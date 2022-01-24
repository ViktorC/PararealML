from pararealml.utils.rand import set_random_seed
from pararealml.utils.tf import use_cpu, use_deterministic_ops

use_cpu()
use_deterministic_ops()
set_random_seed(0)
