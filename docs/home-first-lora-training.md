# Home-first LoRA training

## Direction

Treat owned local compute as the default place to run LoRA training.

The intended system is not "rent the largest cloud GPU and adapt the software to
it." It is closer to a durable home training appliance assembled from inexpensive
or recovered hardware, with software doing as much work as possible to make that
hardware useful.

Cloud GPU capacity remains useful as overflow rather than as the assumed home of
training.

## Hardware priorities

Optimize first for whether a useful training job fits, not for peak advertised
throughput.

In particular:

- prioritize usable accelerator memory before raw compute throughput;
- consider inexpensive used or e-waste machines and accelerators when they can
  provide enough memory for the target model and training state;
- let quantization, LoRA, gradient accumulation, checkpointing, and efficient
  kernels stretch the hardware rather than assuming abundant rented compute;
- accept longer wall-clock runs when the machine is owned and can run unattended.

The interesting engineering question is how much useful training can be extracted
from cheap commodity or discarded hardware by improving the software around it.

## Reliability before solar

A home training machine should tolerate interruption well.

Training should therefore:

- checkpoint often enough that a power loss does not destroy a long run;
- resume from an explicit durable checkpoint;
- make the last completed durable step visible;
- distinguish successful training progress from merely keeping a process alive.

An ordinary UPS is useful for short outages and controlled shutdown. It should be
treated separately from the solar question.

Do not assume that a UPS plus a solar panel makes continuous training free. First
measure the machine's actual energy consumption under representative training
loads. Only after that measurement should panel, inverter, and battery sizing be
considered.

## Cost model

Do not reduce the comparison to hourly cloud-GPU price.

For local training, keep at least these costs distinct:

- acquisition cost of used or recovered hardware;
- upgrades needed primarily to obtain enough accelerator memory;
- household electricity consumed by real training runs;
- UPS and, if justified by measurements, solar and battery hardware;
- time-to-result;
- maintenance and failure replacement.

For rented training, keep at least these costs distinct:

- accelerator rental;
- persistent storage;
- transfer or egress where applicable;
- idle or setup time that is billed;
- the premium paid for finishing a job quickly.

The local system does not have to beat rented hardware on training time. Its main
advantage may be that experiments can keep running without a visible per-hour
meter.

## Cloud boundary

Keep cloud execution available for exceptional jobs:

- a model or training state does not fit locally;
- a temporary experiment needs substantially more memory;
- finishing quickly matters enough to justify rental cost;
- local hardware is unavailable.

Cloud use should be an explicit escalation, not an architectural assumption.

## What to measure before buying more equipment

Before selecting solar hardware or substantially upgrading the training machine,
record:

1. peak accelerator memory required by representative LoRA jobs;
2. actual wall-clock throughput;
3. average and peak whole-machine power draw during training;
4. energy consumed per completed useful training run;
5. checkpoint size and restart cost;
6. failure/restart behavior during a deliberately interrupted run.

Those measurements can drive later decisions about used accelerators, UPS size,
battery capacity, solar generation, or occasional rented GPU capacity.

## Acceptance direction

A useful local-training acceptance case should eventually establish that a job can:

1. start on the intended low-cost local hardware;
2. make measurable training progress;
3. write a durable checkpoint;
4. survive an interruption;
5. resume from that checkpoint without silently restarting;
6. finish with retained receipts for runtime, memory use, and energy use.

That makes "always going at home" an observable property of the training system
rather than a vague infrastructure goal.
