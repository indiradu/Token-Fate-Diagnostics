Premise A: token convergence does not imply representation convergence

This is likely true, but it may be nearly trivial: even after a token identity stops changing, its hidden state and K/V can continue changing because the surrounding masked context continues evolving.

The meaningful version is:

$$ H_A: P(R_{i,t}=0\mid T_{i,t}=1)>0 $$

where:

$$ T_{i,t} = \mathbf 1[ \arg\max p_{i,s} = \arg\max p_{i,T}, \ \forall s\ge t ] $$

means the predicted token remains stable for the rest of the baseline trajectory, and:

$$ R_{i,t} = \mathbf 1[ \max_{s\ge t,\ell} d(K^\ell_{i,s},K^\ell_{i,t}) + d(V^\ell_{i,s},V^\ell_{i,t}) \le \delta_{KV} ] $$

means its layerwise reference state has genuinely settled.

Do not test merely whether hidden drift is nonzero. Floating-point noise and harmless movement will guarantee that result. Test whether token-stable positions exhibit:

substantial layerwise K/V drift;
drift beyond normal numerical variation;
drift large enough to affect active-token logits or attention outputs.

Also, call this token/prediction stability, not “semantic convergence.” Same-token identity is not equivalent to same meaning.

Premise B: representation convergence does not imply safe locking

As written, this premise mixes two different interventions:

committing the token identity;
freezing the token’s K/V reference.

Those need separate hypotheses.

B1: Commitment safety does not imply reference-freeze safety
$$ H_{B1}: P(Y^{F}\neq Y^{C}\mid Y^{C}=Y^{0})>0 $$

where:

\(Y^0\): baseline result;
\(Y^C\): token committed, but representation fully recomputed;
\(Y^F\): token committed and its layerwise K/V frozen.

This is the most important PhaseLock premise: a token intervention can be harmless while a stale-reference intervention at the same token is harmful.

B2: Observed low drift does not guarantee freeze safety
$$ H_{B2}: P(Y^{F}\neq Y^{C}\mid \widehat D^{KV}_{\text{past}}\le\delta)>0 $$

The word observed matters. If “representation convergence” means that all future K/V states are exactly identical, freezing should be safe in a conventional transformer. The relevant failure mode is:

Low drift over the previous few steps does not guarantee future reference settlement.

Therefore distinguish:

retrospective/oracle future settlement;
online convergence inferred only from past and current states.
Premise C: drift predicts reference-freeze harm

Avoid saying drift “influences” answer quality unless the causal mechanism is established. The safer hypothesis is:

$$ H_C: D^{KV}_{\text{pre-freeze}} \text{ predicts the causal harm of freezing} $$

Test whether freeze-failure probability rises across drift quantiles after controlling for:

confidence and entropy;
posterior KL/JSD;
decoding step and remaining horizon;
position and token type;
local mask ratio;
active context volatility;
attention received from active tokens.

Raw drift may not be sufficient. A token can drift substantially but have almost no influence if active tokens barely attend to it. A stronger candidate signal is:

reference risk≈K/V drift×attention exposure.