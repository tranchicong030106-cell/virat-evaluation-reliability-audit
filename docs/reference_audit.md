# Reference audit

This file records the source check used to clean `refs.bib` for the manuscript.

## Policy

For IEEE/CVF conference papers, the bibliography uses DOI/IEEE/DBLP pagination when it differs from the CVF open-access display pagination. This explains several one-page or proceedings-offset differences and avoids mixing pagination conventions inside one bibliography.

## Verified references

| Key | Verification source | Status |
|---|---|---|
| `austin2009balance` | Wiley, DOI `10.1002/sim.3697` | verified |
| `cao2023benchmark` | CVF / DOI `10.1109/CVPR52729.2023.01953` | verified |
| `furnari2019what` | IEEE DOI `10.1109/ICCV.2019.00635`; DBLP pages 6251--6260 | verified |
| `geirhos2020shortcut` | Nature Machine Intelligence, DOI `10.1038/s42256-020-00257-z` | verified |
| `girdhar2021anticipative` | IEEE DOI `10.1109/ICCV48922.2021.01325`; DBLP pages 13485--13495 | verified |
| `gong2022future` | CVF / DOI `10.1109/CVPR52688.2022.00306` | verified |
| `koh2021wilds` | PMLR 139:5637--5664 | verified |
| `lea2017tcn` | IEEE DOI `10.1109/CVPR.2017.113`; DBLP pages 1003--1012 | verified |
| `northcutt2021label` | NeurIPS Datasets and Benchmarks 2021 proceedings | verified |
| `oh2011virat` | CVPR 2011 / DOI `10.1109/CVPR.2011.5995586` | verified |
| `ojala2010permutation` | JMLR 11(62):1833--1863 | verified |
| `rodin2022untrimmed` | ICIAP 2022 / DOI `10.1007/978-3-031-06433-3_29` | verified |
| `ryoo2011activity` | ICCV 2011 / DOI `10.1109/ICCV.2011.6126349` | verified |
| `vondrick2016anticipating` | CVPR 2016 / DOI `10.1109/CVPR.2016.18` | verified |

## Pagination notes

Three references commonly show different page numbers across the CVF open-access site and the final IEEE/DBLP proceedings metadata:

- Furnari and Farinella, ICCV 2019: CVF display `6252--6261`; IEEE/DBLP `6251--6260`.
- Girdhar and Grauman, ICCV 2021: CVF display `13505--13515`; IEEE/DBLP `13485--13495`.
- Lea et al., CVPR 2017: CVF display `156--165`; IEEE/DBLP `1003--1012`.

The manuscript uses the IEEE/DBLP values consistently. These are pagination-system differences, not different papers.

## Cleanup performed

- removed the duplicate `lea2017tcn` BibTeX entry;
- retained one canonical `lea2017tcn` entry with DOI `10.1109/CVPR.2017.113`;
- removed the unused Sultani et al. reference from the working bibliography;
- checked title, author list, year, venue, and DOI or official proceedings URL for every cited item.
