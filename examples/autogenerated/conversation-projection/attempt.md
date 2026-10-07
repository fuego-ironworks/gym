# Cached-target representation training

Input types: finite Float32 feature and target matrices with equal row counts.
Output: encoder coefficients plus a receipt. Signed E5M3 codes contain nine
logical bits, independently of the buffer layout. Storage precision, product
rounding, reduction precision and Float32 optimizer updates are separate choices.

The primary Idriç Float32-vector probe is preserved in `ConversationVector.idric`.
Compiler ff4d852862a3942592f8ade9afde8d409d9803be rejected `Float32` after the
Prelude/base paths and record capitalization were corrected. The same numerical
boundary is needed here. This exact build failure justifies checked Ithon and
foreign NumPy buffers; it does not establish that every Idriç branch lacks the
type. The first language acceptance is this vector probe running with Float32
preserved, followed by contiguous matrices and numerical-library bindings.

The checked Ithon implementation keeps the Float32/NumPy baseline runnable.
Its E5M3 operands use low-nine-bit UInt16, FP16-position UInt16, or dense nine-bit
storage. Prealigned values are code << 7, so expansion can view binary16 bits.
Direct Float32-to-E5M3 rounding avoids Float16 double rounding. Explicit half
products and bounded chunk reductions avoid assuming a vendor's accumulator.
This explicit half implementation is currently much slower than Float32 BLAS.

Every optimizer update rebuilds its weight cache. Static feature conversion is
inside the training timer. Tests exhaust all 512 layouts, finite round trips,
nearest-even boundaries, signed zero, NaN, padding and length rejection, and
require identical trained coefficients from all three E5M3 layouts.

The GitHub workflow runs a public, standard-runner, synthetic numerical control
and retains trained weights. Its custom-shell entrypoint executes checked Ithon;
the temporary workflow file contains JSON configuration, not Bash. A useful
conversation-quality claim requires takeout-svm's corpus and evaluation evidence.

This workload has no rotations. Do not invent a rotation microbenchmark as
evidence about training. Packing, shifts, expansion, converted caches, matrix
products, reduction and optimizer updates all remain inside the timed workload.

Local 96×64 → 16, 80-step control, three runs: median Float32/Float32 0.930 ms;
E5M3 low16 with Float16 products/Float32 reduction 198.25 ms; FP16-position16
188.14 ms; dense9 211.38 ms. The layout change saved about 5% in this small
control, with identical coefficients. These are disposable x86 container
measurements, not actual private-corpus training or ARM performance. They do not
justify replacing the faster Float32 baseline. CI emits its own timing receipt.
