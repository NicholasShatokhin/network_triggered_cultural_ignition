# Zenodo deposit checklist

Recommended deposit type: **Software** or **Other research output**.

Suggested metadata:

- Title: *Network-triggered cultural ignition as a candidate Great Filter: code and data*
- Creator: Mykola Shatokhin
- Affiliation: National University "Kyiv Aviation Institute"
- ORCID: 0000-0003-0028-6208
- Contact: n.shatokhin@gmail.com
- Version: 1.0.0
- Access: Open
- Licence: select **Other (Open)** for the mixed repository and retain the per-file licences in `LICENSE`, `LICENSE-DATA.md` and `docs/THIRD_PARTY_NOTICES.md`.
- Keywords: astrobiology; Great Filter; Fermi paradox; cumulative culture; cultural evolution; agent-based model; archaeology; social networks.

Upload the complete repository ZIP. After Zenodo assigns a DOI:

1. after the GitHub release exists, optionally add its URL as `repository-code` in `CITATION.cff`;
2. replace the repository-availability sentence in `paper/seraj/manuscript_seraj.tex` with the DOI URL;
3. add the DOI URL to the submission e-mail;
4. create a new version if code or data change after peer review rather than overwriting the deposited version.

Do not add private correspondence, reviewer comments, credentials, API keys, temporary logs, build caches or unrelated project archives.
