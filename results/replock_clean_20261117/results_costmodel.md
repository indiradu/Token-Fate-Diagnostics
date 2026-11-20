# Cost-model prediction of speed (Study 5 engine arms)
Time per call = F + c x (mean rows over layers); F, c from each model's cost benchmark. 'FLOP-proportional' = rows only (F = 0), scaled to match the dense decode.
Speed = dense Conf-0.9 time / arm time (sum over prompts). Error = predicted / measured speed - 1.

| model | set | arm | measured speed | cost-model speed (error) | FLOP-proportional speed (error) | prompts |
|---|---|---|---|---|---|---|
| LLaDA-8B | he164 | R0 | 1.58x | 1.45x (-8%) | 2.35x (+49%) | 164 |
| LLaDA-8B | he164 | CTL | 1.65x | 1.55x (-6%) | 2.88x (+75%) | 164 |
| LLaDA-8B | he164 | ELASTIC | 1.32x | 1.37x (+4%) | 2.55x (+93%) | 164 |
| LLaDA-8B | kod_held | R0 | 1.95x | 1.74x (-11%) | 3.33x (+71%) | 150 |
| LLaDA-8B | kod_held | CTL | 1.93x | 1.75x (-9%) | 3.64x (+89%) | 150 |
| LLaDA-8B | kod_held | ELASTIC | 1.63x | 1.64x (+1%) | 3.42x (+110%) | 150 |
| LLaDA-8B | gsm_held | R0 | 1.36x | 1.42x (+4%) | 3.58x (+163%) | 100 |
| LLaDA-8B | gsm_held | CTL | 1.35x | 1.43x (+6%) | 5.23x (+289%) | 100 |
| LLaDA-8B | gsm_held | ELASTIC | 1.16x | 1.42x (+23%) | 4.90x (+324%) | 100 |
| LLaDA-8B | math108 | R0 | 1.98x | 1.85x (-7%) | 4.86x (+146%) | 108 |
| LLaDA-8B | math108 | CTL | 1.99x | 1.84x (-7%) | 5.58x (+181%) | 108 |
| LLaDA-8B | math108 | ELASTIC | 1.76x | 1.87x (+6%) | 6.22x (+253%) | 108 |
| LLaDA-1.5 | he164 | R0 | 1.61x | 1.45x (-10%) | 2.38x (+48%) | 164 |
| LLaDA-1.5 | he164 | CTL | 1.65x | 1.52x (-8%) | 2.83x (+71%) | 164 |
| LLaDA-1.5 | he164 | ELASTIC | 1.34x | 1.36x (+1%) | 2.61x (+95%) | 164 |
| LLaDA-1.5 | kod_held | R0 | 1.97x | 1.72x (-13%) | 3.47x (+77%) | 150 |
| LLaDA-1.5 | kod_held | CTL | 1.98x | 1.76x (-11%) | 3.79x (+91%) | 150 |
| LLaDA-1.5 | kod_held | ELASTIC | 1.65x | 1.63x (-1%) | 3.61x (+118%) | 150 |
| LLaDA-1.5 | gsm_held | R0 | 1.36x | 1.40x (+3%) | 3.56x (+162%) | 100 |
| LLaDA-1.5 | gsm_held | CTL | 1.37x | 1.43x (+4%) | 4.71x (+244%) | 100 |
| LLaDA-1.5 | gsm_held | ELASTIC | 1.18x | 1.42x (+20%) | 4.94x (+319%) | 100 |
| LLaDA-1.5 | math108 | R0 | 1.81x | 1.73x (-4%) | 4.71x (+160%) | 108 |
| LLaDA-1.5 | math108 | CTL | 1.84x | 1.75x (-5%) | 5.25x (+186%) | 108 |
| LLaDA-1.5 | math108 | ELASTIC | 1.66x | 1.82x (+9%) | 6.09x (+266%) | 108 |
| Dream | he164 | R0 | 1.55x | 1.54x (-1%) | 2.00x (+29%) | 164 |
| Dream | he164 | CTL | 1.61x | 1.64x (+2%) | 2.36x (+46%) | 164 |
| Dream | kod_held | R0 | 1.67x | 1.61x (-4%) | 2.23x (+33%) | 150 |
| Dream | kod_held | CTL | 1.75x | 1.74x (-1%) | 2.61x (+49%) | 150 |
| Dream | gsm_held | R0 | 1.42x | 1.56x (+10%) | 3.69x (+161%) | 100 |
| Dream | gsm_held | CTL | 1.45x | 1.64x (+13%) | 5.04x (+247%) | 100 |
| LLaDA-MoE | he164 | R0 | 1.28x | 1.40x (+10%) | 2.23x (+75%) | 164 |
| LLaDA-MoE | he164 | CTL | 1.32x | 1.47x (+11%) | 2.59x (+95%) | 164 |
| LLaDA-MoE | kod_held | R0 | 1.42x | 1.61x (+13%) | 2.88x (+102%) | 150 |
| LLaDA-MoE | kod_held | CTL | 1.48x | 1.69x (+14%) | 3.41x (+130%) | 150 |
| LLaDA-MoE | gsm_held | R0 | 1.23x | 1.39x (+13%) | 3.64x (+196%) | 100 |
| LLaDA-MoE | gsm_held | CTL | 1.24x | 1.43x (+16%) | 5.01x (+306%) | 100 |
| LLaDA-MoE | math108 | R0 | 1.50x | 1.78x (+18%) | 5.39x (+258%) | 108 |
| LLaDA-MoE | math108 | CTL | 1.45x | 1.75x (+20%) | 5.69x (+292%) | 108 |

Mean absolute error over 38 arm-cells: cost model 8.6%, FLOP-proportional 149.9%.