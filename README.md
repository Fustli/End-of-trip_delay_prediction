# End-of-trip delay prediction

## 1. A projekt célja
A projekt célja egy olyan predikciós rendszer létrehozása, amely képes megbecsülni a budapesti buszjáratok várható késését a végállomásra érkezéskor. A feladat során a statikus menetrendi adatokat (GTFS) és a valós idejű járműkövetési adatokat (FUTÁR API) ötvözzük.  
A megoldás során a teljes közlekedési hálózatot gráfként modellezzük, és erre építve fejlesztünk különböző komplexitású modelleket, az egyszerű heurisztikáktól egészen a gráf neurális hálókig (GNN). A végső cél az eredmények összehasonlítása és annak megállapítása, hogy a komplexebb GNN modellek nyújtanak-e szignifikáns előnyt az egyszerűbb megközelítésekkel szemben a késések előrejelzésében.

## 2. Adatforrások és adatgyűjtés

### Források
1. **Statikus GTFS (General Transit Feed Specification)**:  
   Ez a csomag tartalmazza a teljes budapesti közlekedési hálózat statikus adatait: a járatok útvonalait, a megállók helyzetét és a hivatalos menetrendeket. Ez adja a gráfunk vázát.  
   **Elérhetőség**: [BKK GTFS Adatok](https://www.bkk.hu)
   
2. **BKK FUTÁR API**:  
   Ez a valós idejű interfész szolgáltatja a járművek aktuális pozícióját, sebességét és a menetrendhez viszonyított késését másodpercben. Ezek a dinamikus adatok adják a modellünk bemeneti jellemzőit.  
   **Elérhetőség**: [BKK Open Data](https://data.bkk.hu)  
   **Fontos**: Az API használatához ingyenes regisztráció és API kulcs igénylése szükséges.

### Adatgyűjtési folyamat
A projekt alapját egy robusztus, idősoros adatbázis képezi.  
- **Ajánlott gyűjtési időszak**: Minimum 2 hét.  
- **Gyakoriság**: Percenkénti lekérdezés a FUTÁR API-ból a releváns járatok pozíciójára.  
- **Feladat**: Egy adatgyűjtő szkript írása (pl. Pythonban), amely a megadott időközönként lekérdezi és elmenti a járművek aktuális adatait (járműazonosító, járat, útvonal, pozíció, aktuális késés stb.) egy strukturált formátumba (pl. CSV fájlokba vagy egy adatbázisba).

## 3. Modellezési megközelítés
A problémát gráf-alapú predikcióként fogjuk fel.

### Gráf reprezentáció
- **Csomópontok (Nodes)**: A megállók a gráf csomópontjai.
- **Élek (Edges)**: A megállók közötti közvetlen összeköttetések egy adott járat útvonalán.  
  Az élek tulajdonságai lehetnek például a menetrend szerinti utazási idők.

### Modellek
1. **Baseline heurisztikák**:  
   Egyszerű alapmodellek, amelyekkel összevethetjük a GNN teljesítményét. Például:
   - A jelenlegi késés lesz a végső késés.
   - Az adott járat átlagos késése az adott napszakban.

2. **Gráf neurális háló (GNN)**:  
   Egy olyan modell, ahol a csomópontok (megállók) információt cserélnek a szomszédaikkal. Ez lehetővé teszi, hogy a hálózat korábbi szakaszain tapasztalt anomáliák (pl. dugó, torlódás) hatását a modell figyelembe vegye a későbbi megállókra, így a végállomásra vonatkozó predikció során is.

## 4. Adatstruktúra javaslat
Az adatgyűjtés során érdemes a letöltött adatokat strukturáltan, például több CSV fájlban vagy egy relációs adatbázisban tárolni.

### Javasolt fájlok/táblák:
- **GTFS adatok**: A letöltött GTFS csomagból kinyert `trips.txt`, `stops.txt`, `stop_times.txt` stb. fájlok.
- **Valós idejű adatok**: Egy `vehicle_updates.csv` fájl, ami a FUTÁR API-ból percenként gyűjtött adatokat tartalmazza az alábbi oszlopokkal:
  - `timestamp` (a lekérdezés időpontja)
  - `trip_id` (a járat azonosítója)
  - `vehicle_id` (a jármű azonosítója)
  - `last_stop_id` (az utoljára érintett megálló azonosítója)
  - `delay_seconds` (az aktuális késés másodpercben)
  - `latitude`, `longitude` (a jármű pozíciója)

## 5. Adatfeltöltés és felhasználási jogok
A feldolgozott, tanításra kész adatállományt és a projekthez tartozó kódot az alábbi helyre kell feltölteni:

**URL**: [https://bit.ly/bme-dl-pw-2025](https://bit.ly/bme-dl-pw-2025)

### Feltöltés menete:
1. Nyisd meg a fenti linket.
2. Keresd meg az `endoftripdelayprediction` nevű könyvtárat.
3. A könyvtáron belül hozz létre egy új alkönyvtárat a saját Neptun kódoddal.
4. Ebbe a könyvtárba töltsd fel a feldolgozott adatokat, a kódokat és az eredményeket bemutató dokumentációt.

**Fontos**: A feltöltéssel lemondasz az adatokkal kapcsolatos minden jogodról, és hozzájárulsz, hogy az általad gyűjtött és feldolgozott adatokat, valamint a kódokat bárki szabadon felhasználhassa.
