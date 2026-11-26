# Rekonstrukcja i animacja geometrii 3D

Celem zajęć jest odtworzenie powierzchni 3D w postaci siatki trójkątów (mesha) z różnych reprezentacji geometrii oraz wykorzystanie tej siatki do animacji. W pierwszej części próbkujemy analityczne funkcje odległości ze znakiem (Signed Distance Function, SDF). Odtwarzamy kontury metodą Marching Squares i powierzchnie metodą Marching Cubes. Na prostych figurach i bryłach sprawdzamy, jak interpolacja, wybór poziomu pola i rozdzielczość próbkowania wpływają na dokładność, liczbę trójkątów i koszt rekonstrukcji.

W drugiej części rekonstruujemy powierzchnię z syntetycznej chmury punktów, map głębi, pola gęstości NeRF oraz głębi szacowanej z Gaussian Splatting. Porównujemy Ball Pivoting i rekonstrukcję Poissona dla danych z szumem i brakującym fragmentem powierzchni. Zestawiamy triangulację pojedynczej mapy głębi z integracją wielu widoków metodą TSDF Fusion i sprawdzamy, jak próg gęstości NeRF oraz błędy estymacji głębi wpływają na wynikowy mesh.

W ostatniej części deformujemy siatkę chorągiewki i przenosimy próbki jej powierzchni za pomocą współrzędnych barycentrycznych. Porównujemy interpolację położeń odkształconych wierzchołków z bezpośrednią deformacją punktu. Po zajęciach uczestnik powinien umieć dobrać metodę rekonstrukcji do danych wejściowych, ocenić wynik na podstawie wykresów i siatek oraz animować punkty związane z powierzchnią mesha.

**Zawartość repozytorium**

- [Wspólna prezentacja (PPTX)](prezentacja.pptx) oraz [wersja do czytania (PDF)](prezentacja.pdf) obejmują podstawy rekonstrukcji i animacji oraz przykłady wyników. Animacje osadzone w PPTX są w PDF widoczne jako pojedyncze klatki.
- [01_sdf_marching_cubes.ipynb](01_sdf_marching_cubes.ipynb) zawiera demonstracje pól SDF, Marching Squares i Marching Cubes oraz zadanie o rekonstrukcji sumy dwóch kul.
- [02_surface_reconstruction.ipynb](02_surface_reconstruction.ipynb) zawiera rekonstrukcję z chmury punktów, map głębi, NeRF i Gaussian Splatting, zadania z estymacji głębi i triangulacji oraz opcjonalny eksport mesha.
- [03_animacja_meshu_i_chmury.ipynb](03_animacja_meshu_i_chmury.ipynb) zawiera deformację chorągiewki, animację chmury punktów na jej powierzchni oraz zadanie z interpolacji barycentrycznej.
- Moduły [marching_squares.py](scripts/utils/marching_squares.py) i [surface_reconstruction.py](scripts/utils/surface_reconstruction.py) udostępniają odpowiednio logikę rekonstrukcji konturu w komórce oraz pomocnicze funkcje kamery, renderowania głębi i wizualizacji 3D.
- [scripts/generate_figures/](scripts/generate_figures/) zawiera skrypty przygotowujące wybrane grafiki i animacje oraz skrypt treningu modelu NeRF. Gotowe grafiki są w [images/](images/), a wagi modelu w [data/torus_nerf_weights.npz](data/torus_nerf_weights.npz).
- [requirements.txt](requirements.txt) określa zależności projektu.


## Konfiguracja środowiska

Wybierz `venv` albo Condę.

**venv w Linux/macOS**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**venv w Windows (PowerShell)**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Conda**

```bash
conda create -n .venv python=3.12
conda activate .venv
```

Po aktywacji wybranego środowiska zainstaluj pakiety i uruchom JupyterLab:

```bash
python -m pip install -r requirements.txt
jupyter lab
```

## Zadania

**Ćwiczenia na zajęciach**

1. **Czy siatka zauważy mały okrąg?** ([notebook 01, sekcja 3](01_sdf_marching_cubes.ipynb#cwiczenie-maly-okrag))
2. **Próg gęstości NeRF** ([notebook 02, sekcja 5](02_surface_reconstruction.ipynb#prog-gestosci-nerf))

**Dwa zadania obowiązkowe do domu**

1. **Połącz dwie kule** ([notebook 01](01_sdf_marching_cubes.ipynb))
2. **Głębokość piksela** ([notebook 02](02_surface_reconstruction.ipynb))

**Zadania dodatkowe do domu**

- **Triangulacja mapy głębi** ([notebook 02](02_surface_reconstruction.ipynb))
- **Punkt na deformowanym trójkącie** ([notebook 03](03_animacja_meshu_i_chmury.ipynb))

## Źródła

Daty w nawiasach oznaczają weryfikację źródeł podczas opracowywania README.

Repozytorium zostało stworzone na podstawie:<br />
[1] W. E. Lorensen, H. E. Cline (1987). [Marching Cubes: A High Resolution 3D Surface Construction Algorithm](https://www.cs.columbia.edu/~allen/PHOTOPAPERS/marching.cubes.pdf). SIGGRAPH, s. 163–169. DOI `10.1145/37402.37422` (dostęp 25.11.2026).<br />
[2] G. M. Nielson, B. Hamann (1991). [The Asymptotic Decider: Resolving the Ambiguity in Marching Cubes](https://web.cs.ucdavis.edu/~hamann/pubs91.shtml). IEEE Visualization, s. 83–91. DOI `10.1109/VISUAL.1991.175782` (dostęp do wykazu autora 25.11.2026).<br />
[3] A. Doi, A. Koide (1991). [An Efficient Method of Triangulating Equi-Valued Surfaces by Using Tetrahedral Cells](https://globals.ieice.org/en_transactions/information/10.1587/e74-d_1_214/_p). IEICE Transactions on Information, E74-D(1), s. 214–224 (dostęp 25.11.2026).<br />
[4] B. Curless, M. Levoy (1996). [A Volumetric Method for Building Complex Models from Range Images](https://graphics.stanford.edu/papers/volrange/). SIGGRAPH (dostęp 25.11.2026).<br />
[5] B. Mildenhall i in. (2020). [NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis](https://www.matthewtancik.com/nerf). ECCV. [Przykład ekstrakcji mesha udostępniony przez autorów](https://github.com/bmild/nerf/blob/master/extract_mesh.ipynb) (dostęp 25.11.2026).<br />
[6] B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis (2023). [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). ACM Transactions on Graphics (dostęp 25.11.2026).<br />
[7] scikit-image. Dokumentacja [Marching Cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.marching_cubes) i przykład [Contour finding (Marching Squares)](https://scikit-image.org/docs/stable/auto_examples/edges/plot_contours.html) (dostęp 25.11.2026).<br />
[8] Open3D. [Surface reconstruction](https://www.open3d.org/docs/latest/tutorial/geometry/surface_reconstruction.html) oraz [RGBD integration](https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html) (dostęp 25.11.2026).<br />
[9] M. Pharr, W. Jakob, G. Humphreys. *Physically Based Rendering: From Theory to Implementation*. [Triangle Meshes](https://pbr-book.org/4ed/Shapes/Triangle_Meshes), wyd. 4, oraz [Sampling a Triangle](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations#SamplingaTriangle), wyd. 3 (dostęp 25.11.2026).<br />

Warto zwrócić uwagę na:<br />
[10] BorisTheBrave (2018). [Marching Cubes Tutorial](https://www.boristhebrave.com/2018/04/15/marching-cubes-tutorial/) oraz [Marching Cubes 3D Tutorial](https://www.boristhebrave.com/2018/04/15/marching-cubes-3d-tutorial/). Ilustracje do slajdów „Marching Squares: 16 konfiguracji narożników” i „Marching Cubes: przejście do sześcianów” oraz materiał do notebooka 01 (dostęp 25.11.2026).<br />
[11] V. Koutlis (2025). [What is a Point Cloud? Basics, Uses and Visualization](https://rebusfarm.net/blog/what-is-a-point-cloud). RebusFarm. Ilustracja na slajdzie „Chmura punktów jako reprezentacja” jest przypisana w artykule do Packt (dostęp 25.11.2026).<br />
[12] F. Bernardini, J. Mittleman, H. Rushmeier, C. Silva, G. Taubin (1999). [The Ball-Pivoting Algorithm for Surface Reconstruction](https://kogs-www.informatik.uni-hamburg.de/~tchernia/SR_papers/bernardini99.pdf). IEEE Transactions on Visualization and Computer Graphics, 5(4), s. 349–359. Slajd „Ball Pivoting Algorithm” odsyła do [ilustracji „Basic Operations” w SlideServe](https://image2.slideserve.com/4596010/basic-operations-l.jpg) (dostęp 25.11.2026).<br />
[13] M. Kazhdan, M. Bolitho, H. Hoppe (2006). [Poisson Surface Reconstruction](https://hhoppe.com/proj/poissonrecon/). Symposium on Geometry Processing. Ilustrację ze slajdu „Poisson Surface Reconstruction” wskazano w [omówieniu w SlideToDoc](https://slidetodoc.com/poisson-surface-reconstruction-m-kazhdan-m-bolitho-h/) (dostęp 25.11.2026).<br />
[14] U.-J. Kim (2024). [4. 3D Computer Vision](https://uijinee.github.io/posts/3dvision/). UiJin’s Blog. Źródło wskazane przy ilustracji na slajdzie „Mapa głębi (Depth Map) - przypomnienie” (dostęp 25.11.2026).<br />
[15] A. Guédon, V. Lepetit (2024). [SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering](https://anttwo.github.io/sugar/). CVPR. Metoda omawiana na slajdzie „3DGS, SuGaR i 2DGS na danych DTU” (dostęp 25.11.2026).<br />
[16] B. Huang, Z. Yu, A. Chen, A. Geiger, S. Gao (2024). [2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://arxiv.org/abs/2403.17888). SIGGRAPH. Metoda omawiana na slajdzie „3DGS, SuGaR i 2DGS na danych DTU” (dostęp 25.11.2026).<br />
[17] Open3D. [Pakiety dystrybucyjne wersji 0.20.0](https://pypi.org/project/open3d/0.20.0/#files). PyPI (dostęp 25.11.2026).<br />
[18] J. Wong (2016). [Ray Marching and Signed Distance Functions](https://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/) (dostęp 25.11.2026).<br />
[19] D. Bommes. [Geometry Representations](https://slides.cgg.unibe.ch/GP20/03_Geometry_Representations.html#/). Slajdy z wykładu *3D Geometry Processing*, Uniwersytet w Bernie (dostęp 25.11.2026).<br />
[20] Open3D. [Distance Queries](https://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html) (dostęp 25.11.2026).<br />
[21] R. Schmidt (2017). [Merging Meshes with Signed Distance Fields](https://www.gradientspace.com/tutorials/2017/11/21/signed-distance-fields-tutorial). gradientspace (dostęp 25.11.2026).<br />
[22] scikit-image. [Marching Cubes](https://scikit-image.org/docs/stable/auto_examples/edges/plot_marching_cubes.html). Przykład ekstrakcji i wizualizacji izopowierzchni (dostęp 25.11.2026).<br />
[23] S. Lague. [Coding Adventure: Marching Cubes](https://www.youtube.com/watch?v=M3iI2l0ltbE). YouTube (dostęp 25.11.2026).<br />

[24] W. Zhang, Y.-S. Liu, Z. Han (2024). [Neural Signed Distance Function Inference through Splatting 3D Gaussians Pulled on Zero-Level Set](https://papers.neurips.cc/paper_files/paper/2024/file/b8b34328ae68ccc931265bcd21ae5f16-Paper-Conference.pdf). NeurIPS. Rysunek 3 zawiera porównanie na danych DTU przywołane w prezentacji i notebooku 02 (dostęp 26.11.2026).<br />
