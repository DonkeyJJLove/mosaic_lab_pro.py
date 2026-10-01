# 🧩 Mosaic Structure Lab (AST → 3D Honeycomb)

**Mosaic Structure Lab** to interaktywne laboratorium wizualizacji i nauczania struktur mozaikowych, w którym kod źródłowy w języku Python przekształcany jest w trójwymiarową geometrię plastra miodu. Aplikacja pozwala nie tylko zobaczyć, jak z pojedynczych instrukcji i gałęzi drzewa składniowego (AST) wyłania się przestrzeń modularnych komórek, ale także jak — dzięki parametrowi abstrakcji `λ` — można płynnie przechodzić od poziomu mikro (szczegółowe operacje) do poziomu makro (zgrupowane moduły i relacje między nimi).

W ten sposób uczy się myśleć o kodzie nie jako o płaskim tekście, lecz jako o **ontologicznej strukturze**, w której logika systemów, teoria grafów i geometria naturalna (hexagonalna, honeycomb) spotykają się w jednej wspólnej reprezentacji. Tak zbudowana mozaika staje się zarówno narzędziem badawczym, jak i dydaktycznym: pokazuje, jak działa ewolucja topologii w systemach złożonych, jak wizualizować transformacje ontologiczne (TO), oraz jak człowiek i agent-AI mogą korzystać ze wspólnej przestrzeni symboliczno-geometrii do wymiany wiedzy.

---

## ✨ Funkcjonalności

* **Analiza kodu Python (AST)** – parser buduje graf zależności pomiędzy elementami kodu (funkcje, instrukcje, wyrażenia).
* **Wizualizacja w 3D** – węzły i krawędzie są rozmieszczane w przestrzeni zgodnie z regułami mozaiki (osie S i przekątne H). Obsługiwane są transformacje generatywne sterowane parametrem `λ`.
* **Ewolucja topologiczna**:

  * małe `λ` → pełny detal (pojedyncze węzły i krawędzie),
  * duże `λ` → **supergraf**: grupy kontraktują się w centroidy, a między nimi powstają **super-krawędzie**,
  * proces odwzorowuje zasadę **mozaiki DARPA (mosaic/honeycomb)**.
* **Interaktywne sterowanie**:

  * suwak `λ` (interpolacja detal ↔ abstrakcja),
  * checkboxy (widoczność krawędzi S/H, etykiet, centroidów, obrysów grup),
  * parametry (rozmiar punktów, grubość linii, głębokość drzewa, limit elementów),
  * opcja **Ewolucja topologii**,
  * próg intra-group (`λ*`).
* **Stabilna geometria** – wbudowane wyrównanie proporcji pudełka 3D gwarantuje, że obraz nie „oddycha” przy zmianie `λ`. Percepcyjnie zmienia się **topologia**, a nie zoom.

---

## 🚀 Instalacja i uruchomienie

### Wymagania

* Python **3.9+**
* Biblioteki:

  * `tkinter` (zwykle wbudowane),
  * `matplotlib`,
  * `networkx`,
  * *(opcjonalnie)* `numpy` dla warstwy heksów w Meta-świecie.

Instalacja zależności:

```bash
pip install matplotlib networkx
# (opcjonalnie) pip install numpy
```

Uruchomienie:

```bash
python mosaic_lab_pro.py
```

---

## Spis treści

* [Wymagania i uruchomienie](#🚀-instalacja-i-uruchomienie)
* [Przegląd zakładek](#przegląd-zakładek)
* [Architektura mozaiki S/H i TO](#architektura-mozaiki-sh-i-to)
* [Arytmetyka heksów i heurystyka A\*](#arytmetyka-heksów-i-heurystyka-a)
* [Renderowanie przez abstrakcję (λ) i supergraf TO](#renderowanie-przez-abstrakcję-λ-i-supergraf-to)
* [Inwarianty, walidacja i testy naukowe](#inwarianty-walidacja-i-testy-naukowe)
* [Sterowanie i skróty](#sterowanie-i-skróty)
* [Struktura kodu i zależności](#struktura-kodu-i-zależności)
* [Zastosowania: Agent-AI i Human-AI](#zastosowania-agent-ai-i-human-ai)
* [Rozszerzenia i dalsza praca](#rozszerzenia-i-dalsza-praca)
* [Licencja](#licencja)

---

## Przegląd zakładek

1. **Mosaic Lab+** – manualne konstrukcje S/H, planowanie A\*, zapis PNG.
2. **Meta-świat (TO)** – siatka BCC/TO w „wireframe”.
3. **Przekroje 2D** – cięcia XY/YZ/ZX.
4. **Symulacje (S/H)** – budowa artefaktów z klawiatury, wymuszony parytet.
5. **AST Lab** – graf kodu jako mozaika: syntaks (S), semantyka (H), abstrakcja (λ), supergraf TO.

---

## Architektura mozaiki S/H i TO

Scena działa w kratownicy:

$$
\mathcal{L}_2=\{(x,y,z)\in\mathbb{Z}^3:\ x\equiv y\equiv z\pmod 2\}.
$$

* **Porty S (6)**: `(±2,0,0)`, `(0,±2,0)`, `(0,0,±2)`
* **Porty H (8)**: `(±1,±1,±1)`

Warstwa TO wizualizuje **centra komórek** BCC/TO oraz ich kwadraty i heksy w przekrojach.

---

## Arytmetyka heksów i heurystyka A\*

* **Krok S** redukuje \$L1 \leq 2\$, koszt \$\rho\_S=\frac{w\_S}{2}\$.
* **Krok H** redukuje \$L1 \leq 3\$, koszt \$\rho\_H=\frac{w\_H}{3}\$.
* Heurystyka A\* koryguje parytet i zachowuje admissibility/consistency.

Planer `astar` używa 14 sąsiadów i walidacji (`validate_path`).

---

## Renderowanie przez abstrakcję (λ) i supergraf TO

* **S**: parent→child, sibling→next.
* **H**: Use→Def, arg→Func.
* Grupowanie: wg głębokości, typu lub Use-Def.
* Interpolacja:

$$
p_\lambda(n)=\mathrm{lerp}\big(p_{\text{detal}}(n),\ \text{anchor}(n),\ \lambda\big).
$$

* \$\lambda=0\$ → detal,
* \$\lambda=1\$ → komórki i mosty (supergraf TO).

---

## Inwarianty, walidacja i testy naukowe

* Parytet \$x\equiv y\equiv z\$.
* Legalność krawędzi (S/H).
* Ciągłość λ (brak skoków).
* Stabilny box 3D.
* Raport heurystyki (`empirical_heuristic_report`) = brak naruszeń admissibility.

---

## Sterowanie i skróty

**Symulacje (S/H):**

* Strzałki – ruchy osiowe (S).
* `Alt` + strzałki – ruchy ukośne (H).
* `F1–F3` – wybór osi dominującej.
* Spacja – reset.

**AST Lab:**

* Edytuj kod po lewej, `Ctrl+Enter`.
* Suwak `λ`, próg `λ*`, tryb grupowania.
* Opcje: centroidy, bbox, limity głębokości.

---

## Struktura kodu i zależności

* **Geometria**: `H_VECS`, `S_VECS`, `edge_type`.
* **Planer**: `Laws`, `astar`.
* **Scena 3D**: `Scene3D`.
* **Meta-świat (TO)**: kwadraty i heksy.
* **Symulacje**: `InputRouter`.
* **AST Lab**: `AstGraph`, `compute_meta_layers`, `build_supergraph`, `draw_edges_evolving`.

---

## Zastosowania: Agent-AI i Human-AI

* **Wspólna geometria**: ten sam model 14-portowy.
* **Skalowanie poznawcze**: parametr `λ` = „zoom kognitywny”.
* **Mosty semantyczne**: Use→Def i arg→Func jako „belki H”.
* **Supergraf TO**: warstwa do eksploracji i XAI.
* **Bezpieczeństwo**: parytet + admissible heurystyka → brak błędów geometrycznych.

---

## Licencja

Apache License 2.0
Copyright © 2025

