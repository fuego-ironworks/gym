#!/usr/bin/env python3
"""Offline experiment identities, receipt checks and paid-preflight contract audit.

No provider execution or authorization issuer exists in this module. A successful
contract audit is NOT permission to spend. Decimal parameters use JSON strings.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import stat
import sys
import time
from pathlib import Path
from typing import Any

MAX_CASES = 4096
MAX_AGE_SECONDS = 3600
STAGES = ("source-verify", "compatibility", "free-smoke", "resource-estimate",
          "provider-render", "paid-authorize")


class ContractError(ValueError):
    """Fail-closed, machine-readable diagnostic."""


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ContractError(code)


def fields(value: Any, expected: str) -> None:
    require(type(value) is dict and set(value) == set(expected.split()), "FIELDS")


def positive(value: Any, code: str, minimum: int = 1) -> None:
    require(type(value) is int and value >= minimum, code)


def digest_text(value: Any, length: int = 64) -> None:
    require(type(value) is str and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None,
            "DIGEST")


def revision(value: Any) -> None:
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is not None,
            "IMMUTABLE-REVISION-REQUIRED")


def canonical(value: Any) -> bytes:
    def check(item: Any, depth: int = 0) -> None:
        require(depth <= 32, "JSON-DEPTH")
        if type(item) is dict:
            require(all(type(k) is str for k in item), "JSON-KEY")
            for child in item.values():
                check(child, depth + 1)
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        else:
            require(item is None or type(item) in (str, int, bool), "DECIMALS-MUST-BE-STRINGS")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def digest(kind: str, value: Any) -> str:
    return hashlib.sha256(kind.encode("ascii") + b"\0" + canonical(value)).hexdigest()


def load_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            require(key not in result, "DUPLICATE-JSON-KEY")
            result[key] = value
        return result
    def forbidden(_text: str) -> None:
        raise ContractError("DECIMALS-MUST-BE-STRINGS")
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_float=forbidden, parse_constant=forbidden)


def source_ok(source: Any) -> None:
    fields(source, "repository commit dirty")
    require(type(source["repository"]) is str and bool(source["repository"]), "SOURCE")
    revision(source["commit"])
    require(source["dirty"] is False, "DIRTY-SOURCE")


def path_parts(path: Any) -> list[str]:
    require(type(path) is str and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*)*", path)
            is not None, "AXIS-PATH")
    return path.split(".")


def assign(target: dict, path: str, value: Any) -> None:
    parts = path_parts(path)
    for part in parts[:-1]:
        require(part not in target or type(target[part]) is dict, "AXIS-CONFLICT")
        target = target.setdefault(part, {})
    require(parts[-1] not in target, "AXIS-CONFLICT")
    target[parts[-1]] = copy.deepcopy(value)


def lookup(target: dict, path: str) -> Any:
    value = target
    for part in path_parts(path):
        require(type(value) is dict and part in value, "UNKNOWN-EXCLUSION-COORDINATE")
        value = value[part]
    return value


def seal(kind: str, payload: dict) -> dict:
    return {**payload, f"{kind}_id": digest(f"gym-{kind}-v1", payload)}


def check_identity(kind: str, item: dict) -> None:
    require(type(item) is dict and f"{kind}_id" in item, "IDENTITY-MISSING")
    body = {k: v for k, v in item.items() if k != f"{kind}_id"}
    require(item[f"{kind}_id"] == digest(f"gym-{kind}-v1", body), "IDENTITY-MISMATCH")


def validate_case(case: dict) -> None:
    fields(case, "schema source environment_sha256 input_sha256 required_checks coordinates case_id")
    require(type(case["schema"]) is int and case["schema"] == 1, "SCHEMA")
    source_ok(case["source"])
    digest_text(case["environment_sha256"])
    digest_text(case["input_sha256"])
    checks = case["required_checks"]
    require(type(checks) is list and bool(checks) and all(type(x) is str and x for x in checks), "REQUIRED-CHECKS")
    require(len(set(checks)) == len(checks), "DUPLICATE-CHECK")
    require(type(case["coordinates"]) is dict, "COORDINATES")
    model = case["coordinates"].get("model")
    if model is not None:
        fields(model, "repository revision")
        require(type(model["repository"]) is str and bool(model["repository"]), "MODEL")
        revision(model["revision"])
    check_identity("case", case)


def expand(spec: dict) -> list[dict]:
    fields(spec, "schema source environment_sha256 input_sha256 required_checks fixed axes exclude max_cases")
    positive(spec["max_cases"], "MATRIX-LIMIT")
    require(spec["max_cases"] <= MAX_CASES, "MATRIX-LIMIT")
    require(type(spec["fixed"]) is dict and type(spec["axes"]) is dict, "MATRIX")
    require(type(spec["exclude"]) is list, "EXCLUSIONS")
    names = sorted(spec["axes"])
    choices = []
    count = 1
    skeleton = copy.deepcopy(spec["fixed"])
    for name in names:
        # Reject overlapping axes and overrides instead of applying an order.
        assign(skeleton, name, None)
        values = spec["axes"][name]
        require(type(values) is list and bool(values), "EMPTY-AXIS")
        unique = {canonical(v): v for v in values}
        choices.append([unique[key] for key in sorted(unique)])
        count *= len(unique)
        require(count <= spec["max_cases"], "MATRIX-TOO-LARGE")
    for exclusion in spec["exclude"]:
        require(type(exclusion) is dict and bool(exclusion), "EXCLUSION")
        for path in exclusion:
            path_parts(path)
    hit = [False] * len(spec["exclude"])
    cases = {}
    for values in itertools.product(*choices):
        coordinates = copy.deepcopy(spec["fixed"])
        for name, value in zip(names, values):
            assign(coordinates, name, value)
        excluded = False
        for i, rule in enumerate(spec["exclude"]):
            if all(canonical(lookup(coordinates, k)) == canonical(v) for k, v in rule.items()):
                hit[i] = True
                excluded = True
        case = seal("case", {key: spec[key] for key in
                    ("schema", "source", "environment_sha256", "input_sha256", "required_checks")}
                    | {"coordinates": coordinates})
        validate_case(case)
        if excluded:
            continue
        identifier = case["case_id"]
        require(identifier not in cases or canonical(cases[identifier]) == canonical(case), "ID-COLLISION")
        cases[identifier] = case
    require(all(hit), "EXCLUSION-MATCHES-NOTHING")
    require(bool(cases), "EMPTY-MATRIX")
    return [cases[key] for key in sorted(cases)]


def make_job(case: dict, provider: str, resources: dict, request: dict) -> dict:
    validate_case(case)
    job = seal("job", {"schema": 1, "case_id": case["case_id"], "provider": provider,
                       "resources": resources, "request": request})
    validate_job(job, case)
    return job


def validate_job(job: dict, case: dict) -> None:
    validate_case(case)
    fields(job, "schema case_id provider resources request job_id")
    require(type(job["schema"]) is int and job["schema"] == 1, "SCHEMA")
    require(job["case_id"] == case["case_id"], "WRONG-CASE")
    require(job["provider"] in ("local", "github-standard", "vast", "hetzner"), "PROVIDER")
    fields(job["resources"], "ram_bytes disk_bytes runtime_seconds accelerator accelerator_count")
    for key in ("ram_bytes", "disk_bytes", "runtime_seconds"):
        positive(job["resources"][key], "RESOURCE")
    positive(job["resources"]["accelerator_count"], "RESOURCE", 0)
    require(type(job["resources"]["accelerator"]) is str, "RESOURCE")
    require((job["resources"]["accelerator"] == "none") == (job["resources"]["accelerator_count"] == 0), "RESOURCE")
    require(type(job["request"]) is dict, "REQUEST")
    check_identity("job", job)


def all_checks(checks: Any, expected: list[str]) -> None:
    require(type(checks) is dict and set(checks) == set(expected), "CHECK-COVERAGE")
    require(all(value == "pass" for value in checks.values()), "CHECK-NOT-PASSED")


def audit_preflight(case: dict, job: dict, events: list, *, now: int) -> dict:
    """Validate a trace; never authenticate an operator or enable an executor."""
    validate_job(job, case)
    require(job["provider"] in ("vast", "hetzner"), "NOT-PAID-JOB")
    require(job["resources"]["accelerator_count"] == 0, "GPU-PREFLIGHT-NOT-IMPLEMENTED")
    positive(now, "CLOCK")
    require(type(events) is list and len(events) == len(STAGES), "TRACE-INCOMPLETE")
    previous = 0
    for stage, event in zip(STAGES, events):
        fields(event, "stage status job_id case_id source_commit at evidence evidence_sha256")
        require(event["stage"] == stage, "STAGE-ORDER")
        require(event["status"] == "pass", "STAGE-NOT-PASSED")
        require(event["job_id"] == job["job_id"] and event["case_id"] == case["case_id"], "WRONG-JOB")
        require(event["source_commit"] == case["source"]["commit"], "STALE-SOURCE")
        positive(event["at"], "CLOCK")
        require(previous <= event["at"] <= now and now - event["at"] <= MAX_AGE_SECONDS, "STALE-OR-FUTURE-EVIDENCE")
        previous = event["at"]
        require(type(event["evidence"]) is dict, "EVIDENCE")
        require(event["evidence_sha256"] == digest("gym-evidence-v1", event["evidence"]), "EVIDENCE-DIGEST")
    evidence = {e["stage"]: e["evidence"] for e in events}
    source = evidence["source-verify"]
    fields(source, "source")
    require(source["source"] == case["source"], "STALE-SOURCE")
    compatible = evidence["compatibility"]
    fields(compatible, "environment_sha256 compatible")
    require(compatible["compatible"] is True and compatible["environment_sha256"] == case["environment_sha256"], "INCOMPATIBLE")
    smoke = evidence["free-smoke"]
    fields(smoke, "executor run_id run_attempt coverage environment_sha256 result_sha256 checks")
    require(smoke["executor"] == "github-standard", "SMOKE-EXECUTOR")
    require(smoke["coverage"] == "exact", "SURROGATE-NOT-AUTHORITY")
    require(smoke["environment_sha256"] == case["environment_sha256"], "SMOKE-ENVIRONMENT")
    positive(smoke["run_id"], "SMOKE-RUN")
    positive(smoke["run_attempt"], "SMOKE-ATTEMPT")
    digest_text(smoke["result_sha256"])
    all_checks(smoke["checks"], case["required_checks"])
    estimate = evidence["resource-estimate"]
    fields(estimate, "basis measurement_sha256 ram_bytes disk_bytes runtime_seconds")
    require(estimate["basis"] == "measured", "UNMEASURED-RESOURCES")
    digest_text(estimate["measurement_sha256"])
    for key in ("ram_bytes", "disk_bytes", "runtime_seconds"):
        positive(estimate[key], "RESOURCE")
        require(estimate[key] <= job["resources"][key], "RESOURCE-ENVELOPE")
    render = evidence["provider-render"]
    fields(render, "dry_run mutations request_sha256 currency total_microunits expires_at")
    require(render["dry_run"] is True and type(render["mutations"]) is int and render["mutations"] == 0, "NOT-DRY-RUN")
    require(render["request_sha256"] == digest("gym-request-v1", job["request"]), "REQUEST-MISMATCH")
    require(type(render["currency"]) is str and re.fullmatch(r"[A-Z]{3}", render["currency"]) is not None, "CURRENCY")
    positive(render["total_microunits"], "PRICE", 0)
    positive(render["expires_at"], "QUOTE-EXPIRED")
    require(render["expires_at"] > now, "QUOTE-EXPIRED")
    approval = evidence["paid-authorize"]
    fields(approval, "approved operator request_sha256 preflight_sha256 currency max_total_microunits expires_at")
    require(approval["approved"] is True and type(approval["operator"]) is str and bool(approval["operator"].strip()), "NO-APPROVAL")
    require(approval["request_sha256"] == render["request_sha256"], "REQUEST-MISMATCH")
    require(approval["preflight_sha256"] == digest("gym-preflight-v1", events[:-1]), "APPROVAL-BINDING")
    require(approval["currency"] == render["currency"], "CURRENCY-MISMATCH")
    positive(approval["max_total_microunits"], "BUDGET", 0)
    require(render["total_microunits"] <= approval["max_total_microunits"], "OVER-BUDGET")
    positive(approval["expires_at"], "APPROVAL-EXPIRED")
    require(approval["expires_at"] > now, "APPROVAL-EXPIRED")
    return {"contract_valid": True, "case_id": case["case_id"], "job_id": job["job_id"],
            "trace_sha256": digest("gym-trace-v1", events), "execution_enabled": False,
            "reason": "Offline audit only; trusted evidence, operator authority and live lifecycle gate are not implemented."}


def verify_run(root: Path, case: dict, job: dict) -> dict:
    """Independently rehash a completed, run-scoped bundle; no tensor unpickling."""
    validate_job(job, case)
    require(not root.is_symlink() and root.is_dir(), "BUNDLE-ROOT")
    receipt_path = root / "run.json"
    require(not receipt_path.is_symlink() and receipt_path.is_file(), "MISSING-RECEIPT")
    receipt = load_json(receipt_path)
    fields(receipt, "schema job_id case_id status checks artifacts")
    require(type(receipt["schema"]) is int and receipt["schema"] == 1, "SCHEMA")
    require(receipt["job_id"] == job["job_id"] and receipt["case_id"] == case["case_id"], "WRONG-JOB")
    require(receipt["status"] == "complete", "INCOMPLETE-RUN")
    all_checks(receipt["checks"], case["required_checks"])
    require(type(receipt["artifacts"]) is list and bool(receipt["artifacts"]), "EMPTY-ARTIFACTS")
    seen = set()
    for item in receipt["artifacts"]:
        fields(item, "path size_bytes sha256")
        name = item["path"]
        require(type(name) is str and name not in seen and name != "run.json", "ARTIFACT-PATH")
        require(not name.startswith("/") and "\\" not in name and ":" not in name and
                all(part not in ("", ".", "..") for part in name.split("/")), "ARTIFACT-PATH")
        seen.add(name)
        path = root
        for part in name.split("/"):
            path /= part
            require(not path.is_symlink(), "ARTIFACT-SYMLINK")
        require(path.is_file() and stat.S_ISREG(path.stat().st_mode), "ARTIFACT-MISSING")
        positive(item["size_bytes"], "ARTIFACT-SIZE", 0)
        digest_text(item["sha256"])
        hashed = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                hashed.update(block)
        require(size == item["size_bytes"] and hashed.hexdigest() == item["sha256"], "ARTIFACT-CORRUPT")
    actual = set()
    for path in root.rglob("*"):
        require(not path.is_symlink(), "ARTIFACT-SYMLINK")
        if path.is_dir():
            continue
        require(stat.S_ISREG(path.stat().st_mode), "ARTIFACT-TYPE")
        actual.add(path.relative_to(root).as_posix())
    require(actual == seen | {"run.json"}, "UNINDEXED-ARTIFACT")
    return {"verified": True, "job_id": job["job_id"], "artifact_count": len(seen),
            "receipt_sha256": digest("gym-run-v1", receipt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    expand_args = sub.add_parser("expand")
    expand_args.add_argument("spec", type=Path)
    for command in ("audit", "verify-run"):
        p = sub.add_parser(command)
        p.add_argument("case", type=Path)
        p.add_argument("job", type=Path)
        p.add_argument("input", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "expand":
            result = {"schema": 1, "cases": expand(load_json(args.spec))}
        elif args.command == "audit":
            result = audit_preflight(load_json(args.case), load_json(args.job), load_json(args.input), now=int(time.time()))
        else:
            result = verify_run(args.input, load_json(args.case), load_json(args.job))
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    except (ContractError, OSError, ValueError, TypeError, KeyError) as error:
        print(json.dumps({"passed": False, "execution_enabled": False, "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
