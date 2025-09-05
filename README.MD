# 🧩 Mosaic AST Lab — Honeycomb Graph & Python AST Compass 3D

## Opis projektu

**Mosaic AST Lab** to interaktywna aplikacja GUI (Python + Tkinter + Matplotlib), która łączy:

- **Mozaikowy model** oparty o **ośmiościan ścięty** (14 portów: 6 osiowych `S` i 8 skośnych `H`).
- **Analizę kodu Pythona** poprzez **AST** (Abstract Syntax Tree).
- **Kompas 3D** wizualizujący zależności syntaktyczne (`S`) i semantyczne (`H`) w postaci grafu.
- **Pad kierunków** umożliwiający interaktywną eksplorację 14 kierunków mozaiki.

Projekt jest eksperymentem na styku:
- teorii grafów,
- wizualizacji struktur obliczeniowych,
- gamifikacji i intuicyjnej eksploracji kodu źródłowego.

---

## Funkcje

- Edytor kodu Pythona (wbudowane okno tekstowe).
- Automatyczne renderowanie (debounce 600 ms) lub ręczne (`Ctrl+Enter`).
- Przełączniki warstw wizualizacji:
  - `S` — syntaktyka (parent → child, kolejność bloków).
  - `H` — semantyka (Use → Def, argumenty funkcji, przepływ danych).
  - etykiety węzłów AST.
- Ustawienia rozmiaru punktów, grubości krawędzi, limitu węzłów i głębokości Z.
- **Pad kierunków (14 portów)**:
  - `S` — X±, Y±, Z±.
  - `H` — wszystkie 8 kombinacji (±,±,±).
- Funkcje pomocnicze:
  - reset widoku,
  - zapis sceny do pliku PNG,
  - auto-obrót sceny 3D.

---

## Instalacja

### Wymagania
- Python 3.8+
- Tkinter (wbudowany w standardowe dystrybucje Pythona)
- Matplotlib

Instalacja bibliotek:
```bash
pip install matplotlib
