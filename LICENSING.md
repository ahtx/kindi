# Licensing and upstream attribution

This modified Kindi program is distributed under **AGPL-3.0-only**, matching
the SPDX declarations of upstream Kindi. `LICENSE` is unchanged from upstream
`6ac903e5cb00a6d55b6cb60f6ad7445cafe93051`. Original author: Amir Husain.
The fork's modifications and relevant date are prominently listed in README,
NOTICE and MODIFICATIONS.md. No additional restriction on recipients is imposed.

Preserve copyright and license notices, disclaimers and notices of changes.
When conveying this modified program, supply its corresponding source under AGPL.
When users interact with the modified version remotely, prominently offer them
its corresponding source free of charge (AGPL section 13). The Lightning API
implements that offer at `/source` and `/source/archive`, and exposes `/license`.
The downloadable ZIP contains the reviewed source files, build/launch scripts,
tests and license notices for that running build. It excludes private configuration,
credentials, request history, weights and the development workspace.

`SOURCE-FILES.txt` is the explicit archive manifest. After modifying the program,
update the manifest for any added files, run `python3 scripts/check_release.py`,
rebuild and restart. Do not mount a different source tree over a running container
or hide these endpoints behind a proxy. A frontend that consumes this API should
prominently link users to the backend's source offer. If you distribute an image
or executable, accompany it with matching corresponding source and retain all
applicable third-party notices. A source link to unmodified upstream alone is
insufficient for a changed backend.

`COMMERCIAL-LICENSE.md` is the original author's information, preserved verbatim.
It does not grant a commercial exception for this fork's additional contributions.
Contacting the author is not required to use this fork under AGPL.

The model is obtained separately under OpenMDW 1.1. Its notice is retained for
provenance; this source release contains no model weights. Runtime dependencies
retain their own licenses (see NOTICE). NVIDIA's CUDA toolkit is separately
obtained by the source build under NVIDIA's terms. The model and CUDA licenses
must not be represented as AGPL licenses.

Source: https://www.gnu.org/licenses/agpl-3.0.html (sections 4, 5, 6 and 13).
Upstream requirements reviewed: LICENSE, NOTICE, COMMERCIAL-LICENSE.md and README
at the pinned Kindi commit. No separate approval requirement for an AGPL fork was
found in these documents. This is an implementation checklist, not a claim of
endorsement or a legal certification.
