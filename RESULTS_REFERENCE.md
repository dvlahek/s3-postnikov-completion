# Final reference results

Formal/topological checks:
- 1024/1024 generic-vs-explicit ordered-five-simplex Sq^2 matches;
- 32768 compatible (alpha,zeta) assignments;
- 216064 systematic descendant-identity test pairs;
- suspended-Hopf preimage framing winding = 1;
- Spin(3) lift endpoint = -1 to floating-point precision;
- smooth skyrmion and gerbe benchmarks approach fourth-order accuracy.

Sampler validation at L=4:
- beta=0.600: fixed-count Wolff 0.77312 +/- 0.00071; exact heat bath 0.77337 +/- 0.00111; z=-0.19;
- beta=0.610: fixed-count Wolff 0.76131 +/- 0.00073; exact heat bath 0.76140 +/- 0.00108; z=-0.07;
- adaptive measurement-time negative control: z=-5.39 and -6.94.

Final production campaign:
- 124 chains, 31 ensembles, 800 measurements per chain;
- max split-Rhat(xi/L) in beta=0.600--0.610: 1.0044;
- max split-Rhat(action): 1.0114;
- max split-Rhat(m2): 1.0066;
- max |hot-cold z(xi/L)|: 2.04.

Large-volume crossings:
- xi/L: (10,12)=0.6080, (12,16)=0.6080, (10,16)=0.6080;
- Binder Q: corresponding crossings approximately 0.6085.

These are used as finite-size dynamical consistency checks. No precision infinite-volume critical coupling or critical exponent is claimed.
