# Pyetje për Assoc. Prof. Dr. Arban Uka

**Teza:** Segmentim i instancave dhe kuantifikim i defekteve të tipit rruazë në imazhe SEM të membranave nanofibroze të elektrotjerrura
**Studenti:** Erlind Skura — Master, Inxhinieri Kompjuterike, Universiteti Epoka
**Mbrojtja:** Shtator 2026

> **Nëse takimi është i shkurtër, këto 5 janë vendimtare:** 1, 2, 3, 4, 12.

---

## 0. Çfarë kanë zgjidhur tashmë të dhënat

Këto nuk kanë nevojë të pyeten — janë nxjerrë nga vetë skedarët dhe duhen vetëm **konfirmuar**:

| Pyetje e vjetër | Përgjigjja nga të dhënat |
|---|---|
| A ka akses te imazhet? | Po — dosja `Segmentations_VB` |
| A janë anotuar? | Po — LabelMe 5.4.1, poligone, **606 rruaza** |
| Sa imazhe? | **11 origjinale** (jo 143 — 132 janë kopje të shumëfishuara) |
| SEM apo TEM? | SEM, Zeiss, 15.00 kV, WD 8.5–11.0 mm |
| Zmadhime? | 500×, 1000×, 3000× — të përziera |
| Klasa? | Një e vetme: `bead` |
| Shkalla e pikselit? | E nxjerrë nga shiriti i shkallës: **555.6 / 277.8 / 92.6 nm/piksel** |
| A është temë "mjekësore"? | Jo — është karakterizim materialesh (elektrotjerrje) |

---

## 1. Përmasat e punës përballë afatit të Shtatorit ⚠️

**Kjo është pyetja e parë që duhet bërë.** Mbrojtja është në Shtator; janë rreth **7 javë**. Kodi i trajnimit nuk është shkruar ende.

**Cili nga këta tre variante e dëshironi?**

- **A — I sigurt.** Vetëm Mask R-CNN, me validim *leave-one-specimen-out*, plus metoda klasike (Otsu + watershed) si bazë krahasimi. Pa U-Net.
- **B — I plotë.** Të tre metodat: klasike, U-Net semantik, Mask R-CNN. Ky është plani aktual i Kapitullit 3.
- **C — Shtyrje.** Afati shtyhet dhe punohet edhe me modele me prior forme (StarDist, Cellpose).

*Kapitulli 3 aktualisht është shkruar për variantin B. Nëse zgjidhni A, e ngushtoj menjëherë.*

---

## 2. Çfarë janë Z2, Z4, Z5 dhe Z6?

Kjo është boshllëku më i madh në tezë tani. Unë di se janë katër mostra, por jo se **çfarë i dallon**.

- Cili polimer dhe cili tretës?
- Cilët parametra ndryshojnë mes tyre — përqendrimi, voltazhi, rrjedha, distanca kolektor–gjilpërë?
- A janë katër receta të ndryshme, apo e njëjta recetë në katër kushte?

*Pa këtë, Kapitulli 1 nuk shpjegon dot pse ekzistojnë këto mostra, dhe "reagimi ndaj parametrave të procesit" te puna e ardhshme mbetet pa bazë.*

**Pyetje shtesë:** Pse mungon `Z4-1` (500×)? Dhe pse mostra Z6 e ka emrin `Z6-4` në vend të `Z6-3`?

---

## 3. A ka më shumë mostra?

Kjo është shtesa më e vlefshme e mundshme për tezën — **më shumë mostra, jo më shumë imazhe të të njëjtave mostra.**

- Katër mostra japin vetëm katër *folds* validimi. Dhjetë do të jepnin një pretendim shumë më të fortë.
- A ekzistojnë imazhe të tjera **pa anotime**? Edhe të paanotuara janë të përdorshme (vetë-trajnim, gjysmë-i-mbikëqyrur) dhe si test cilësor.

---

## 4. Anotimet

- **Kush i bëri?** Një person apo disa?
- **A mund ta ri-anotojë një person i dytë një nënbashkësi** (p.sh. 2 imazhe)?
  *Kjo do ta kthente kufizimin më të madh të tezës në një madhësi të matur: pajtueshmërinë mes anotuesve. Një model që arrin këtë prag ka arritur kufirin e dobishëm të detyrës.*
- **A janë shteruese?** A është vizatuar **çdo** rruazë në çdo imazh, apo vetëm ato të qarta?
- **Ku është kufiri mes një rruaze dhe një fibre të trashur lokalisht?** A ka një kriter të deklaruar, apo është gjykim i anotuesit?

---

## 5. Konfirmim i kalibrimit fizik

Nga shiriti i shkallës nxora:

| Zmadhimi | Gjerësia e fushës | nm/piksel |
|---|---|---|
| 500× | 568.9 µm | 555.6 |
| 1000× | 284.4 µm | 277.8 |
| 3000× | 94.8 µm | 92.6 |

Duke e zbatuar këtë, diametri mesatar i rruazave del **7.92 / 7.34 / 6.75 µm** në të tre zmadhimet — pra i qëndrueshëm.

- **A ju duken këto përmasa fizikisht të sakta për këtë material?**
  *Nëse po, kalibrimi është i vërtetuar dhe të gjitha rezultatet raportohen në mikrometra.*

---

## 6. Protokolli i vlerësimit dhe kopjet e shumëfishuara

Dy vendime metodologjike që dua t'i miratoni para se të vazhdoj:

**(a)** Të dhënat kanë 143 çifte imazh–anotim, por janë **11 origjinale × 13 variante** (flip, shift, stretch). Unë i kam **përjashtuar** kopjet dhe e bëj shumëfishimin gjatë trajnimit, pas ndarjes.
*Arsyeja: kopjet u krijuan përpara ndarjes, ndaj një ndarje e rastësishme mbi 143 skedarë do të fuste variante të të njëjtit imazh njëkohësisht në trajnim dhe në test. Rezultati do të matte memorizim.*
**A jeni dakord?** (I kam kontrolluar — transformimet janë të sakta, thjesht nuk janë kampione të pavarura.)

**(b)** Meqë 11 imazhet vijnë nga vetëm 4 mostra, ndarja bëhet **sipas mostrës** (*leave-one-specimen-out*), jo sipas imazhit.
**A e miratoni?** A duhet lënë ndonjë mostër plotësisht jashtë si test përfundimtar?

---

## 7. Qëllimi shkencor

- **Cila madhësi e përdor vërtet laboratori** — numri i rruazave, dendësia për mm², shpërndarja e madhësisë, apo forma?
  *Kjo përcakton çfarë raportohet si rezultat kryesor në Kapitullin 5.*
- **Si kryhet tani manualisht dhe sa kohë merr një imazh?**
  *Nëse e krahasoj modelin me një person që punon me ImageJ, ky është rezultati më bindës i tezës.*
- **A ju intereson edhe diametri i fibrave**, apo vetëm rruazat?
  *Nëse po, shtrirja e tezës rritet ndjeshëm — ka rëndësi për pyetjen 1.*

---

## 8. Pragu i suksesit

**Cili numër konkret do ta konsideronit sukses?**
P.sh. "AP₅₀ mbi 0.70", ose "gabim nën 10% te numërimi i rruazave".

*Objektivat e Kapitullit 1 i kam shkruar me kritere të matshme; aty tani ka një vlerë provizore `AP₅₀ ≥ 0.70` që pret miratimin tuaj.*

---

## 9. Zbatimi teknik

- **Nga e para apo me librari të gatshme** (Detectron2, Ultralytics, MMDetection) dhe *backbone* të para-trajnuar?
  *E pyes sepse në teza të mëparshme implementimi nga e para është paraqitur si kontribut. Për segmentimin, *transfer learning* nga një model i para-trajnuar është praktika standarde dhe jep rezultat më të mirë me 11 imazhe.*
- **A ka GPU në laborator?**
  *Mask R-CNN me 4 *folds* është praktikisht i pamundur në CPU.*
- **A e vlerësoni një demo të publikuar** (HuggingFace Spaces) ku laboratori ngarkon një imazh dhe merr numërimin?
- **A duhet kodi në Shtojcën A**, apo mjafton një lidhje GitHub?

---

## 10. Titulli dhe administrata

- **A e miratoni titullin?**
  Aktualisht: *"Deep Learning for Instance Segmentation and Quantification of Bead Defects in SEM Images of Electrospun Nanofibre Mats"*
- **Data e saktë e mbrojtjes në Shtator** — faqja e miratimit kërkon datë të plotë.
- **Dy anëtarët e tjerë të komisionit** — për momentin kam vetëm emrin tuaj.
  *(Përgjegjësi i Departamentit — Dr. Florenc Skuka — është i konfirmuar.)*
- **A ka kufizime konfidencialiteti** për t'i publikuar këto imazhe në tezë ose në një punim?
- **A pritet edhe një punim shkencor?**
- **Cilat nga punimet tuaja** për analizën e imazheve të nanofibrave dëshironi t'i citoj në Kapitullin 2?
  *Nuk dua t'i hamendësoj titujt.*
- **Qëndrimi juaj për përdorimin e AI në shkrim** dhe çfarë duhet deklaruar.
