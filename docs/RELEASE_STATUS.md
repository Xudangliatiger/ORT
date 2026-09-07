# Validation status

Three CPU regression tests passed: ORT forward/backward with ordinal weights,
order schedule, and accumulated mid-epoch checkpoint/resume. Final resumed
weights were bitwise identical to uninterrupted training.

GPU training/resume and real-checkpoint inference are being validated.
Full-dataset training, throughput, and 50k-image metrics have not been validated.
Multiworker crop RNG equivalence across resume is not guaranteed.

AliTok-specific redistribution terms remain unresolved in inspected upstream
revision 15410babdf944c957dcde96cc6d09e246bad9c3e; inherited notices alone do not
establish permission for those adaptations.
