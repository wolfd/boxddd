# Voxel collider provenance

The voxel shape, sparse occupancy queries, collision routines, manifold
integration, and their native tests in this Box3D fork originated as a port and
adaptation of [Tribulla/vox3D](https://github.com/Tribulla/vox3D), audited at
commit `912fd36841217ff57309f7c800e5a19742f023f4`.

The affected source files preserve their existing
`SPDX-FileCopyrightText: 2026 Tribulla` and
`SPDX-License-Identifier: MIT` lines. Substantial subsequent changes by the
BoxDDD maintainer additionally carry
`SPDX-FileCopyrightText: 2026 Danny Wolf`.

Tribulla directly confirmed to the BoxDDD maintainer that the complete vox3D
library is available under the MIT license. At the audited revision, however,
the repository README also described the custom voxel collider as originating
in ThePlasticPotato's KRUNCH_AVBD and called that source ARR, while the root
LICENSE and the library-level representation were MIT. These statements should
be reconciled publicly, and any additional original copyright holder should be
added to this notice when confirmed. The existing attribution is preserved in
the meantime.

The planned canonical voxel-patch workspace was informed by studying
[Dimforge Parry](https://github.com/dimforge/parry) version 0.30.2, commit
`1be4b1a7cd0a090bd7efb1207b7bc0d453f4132e`. Parry is Apache-2.0 licensed.
Parry source code is not included or translated in this fork; the BoxDDD work
uses Parry as an algorithmic design reference and is implemented independently
in Box3D's C architecture. If Parry source or substantial protected expression
is incorporated later, the applicable Apache-2.0 license and notices must be
added before distribution.

