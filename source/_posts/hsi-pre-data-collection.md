---
title: Multi-Spectral Imaging 数据采集前期调研
tags:
  - hsi
  - data
  - camera
  - optics
date: 2020-10-16 16:57:04
---

## Equipment

### FLIR Blackfly S RGB Camera

1. Spectral Range: 

   - Blue: 460 nm
   - Green: 530 nm
   - Red: 625 nm

2. Resolution: 720 × 540

3. FPS: 522

   <!-- more -->

4. Dimensions [W x H x L]: 29 mm × 29 mm × 30 mm

5. Official Link: [Link](https://www.flir.com/products/blackfly-s-usb3/)

### XIMEA MQ022HG-IM-SM5X5-NIR Multispectral Camera

1. Spectral Range: 665~975nm
2. Resolution: 
   - Original: 2048 × 1088 
   - Spatial: 409 × 217
3. FPS: up to 170 cubes/sec
4. Sensor size: 2/3"
5. Dimensions WxHxD: 26 x 26 x 31 mm
6. Pixel size: 5.5 µm
7. Python multispectral processing lib: [Link](http://www.spectralpython.net/#documentation)
8. Camera control official python lib: [Link](https://www.ximea.com/support/wiki/apis/Python)
9. Official brief specification: [Link](https://www.ximea.com/files/brochures/xiSpec-Hyperspectral-cameras-2015-brochure.pdf) 
10. Official Page: [Link](https://www.ximea.com/en/products/hyperspectral-cameras-based-on-usb3-xispec/mq022hg-im-sm5x5-nir)

| Full Specifications:      |                                                  |
| ------------------------- | ------------------------------------------------ |
| **Part Number**           | MQ022HG-IM-SM5X5-NIR                             |
| **Resolution**            | Original: 2048 × 1088 Spatial: 409 × 217         |
| **Frame rates**           | up to 170 cubes/sec                              |
| **Sensor type**           | CMOS, Hyperspectral filters added at wafer-level |
| **Sensor model**          | IMEC SNm5x5                                      |
| **Sensor size**           | 2/3"                                             |
| **Sensor active area**    | 25 Bands                                         |
| **Readout Method**        | Snapshot Mosaic                                  |
| **Pixel size**            | 5.5 µm                                           |
| **ADC -Bits per pixel**   | 8, 10 bit RAW pixel data                         |
| **Data interface**        | USB 3.1 Gen1 or PCI Express (xiX camera model)   |
| **Data I/O**              | GPIO IN, OUT                                     |
| **Power consumption**     | 1.6 Watt                                         |
| **Lens mount**            | C or CS Mount                                    |
| **Weight**                | 32 grams                                         |
| **Dimensions WxHxD**      | 26 x 26 x 31 mm                                  |
| **Operating temperature** | 50 °C                                            |
| **Spectral range**        | 665-975 nm                                       |
| **Customs tariff code**   | 8525.80 30 (EU) / 8525.80 40 (USA)               |
| **ECCN**                  | EAR99                                            |

### Seek Compact Pro Thermal Camera

1. Seek Compact Pro: 7500~14000 nm
2. Resolution:  **320 x 240**
3. Field of view: **32°** 
4. Frame rate: **< 9 Hz** 
5. Focusable lens
6. Platform: Android or iOS (Linux 3rd-party binary library)
7. Specification Sheet: [Link](https://www.thermal.com/uploads/1/0/1/3/101388544/compactpro-sellsheet-website.pdf)



## Introduction

Full spectrum:

![img](2880px-EM_spectrum.svg.png)

![img](2880px-EM_Spectrum_Properties_edit_zh.svg.png)

![img](Atmospheric_electromagnetic_opacity.svg)

![img](200px-Light_spectrum.svg.png)

### **Different Infrared:** 

|                       Division name                        |                         Abbreviation                         |                   Wavelength                    |                          Frequency                           |                   Photon energy                   | Temperature[[i\]](https://en.wikipedia.org/wiki/Infrared#cite_note-†-15) |                       Characteristics                        |
| :--------------------------------------------------------: | :----------------------------------------------------------: | :---------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|                       Near-infrared                        |     NIR, IR-A *[DIN](https://en.wikipedia.org/wiki/DIN)*     | 0.75–1.4 [μm](https://en.wikipedia.org/wiki/Μm) | 214–400 [THz](https://en.wikipedia.org/wiki/Terahertz_(unit)) | 886–1653 [meV](https://en.wikipedia.org/wiki/MeV) | 3,864–2,070 [K](https://en.wikipedia.org/wiki/Kelvin) (3,591–1,797 [°C](https://en.wikipedia.org/wiki/Celsius)) | Defined by water absorption,[*[clarification needed](https://en.wikipedia.org/wiki/Wikipedia:Please_clarify)*] and commonly used in [fiber optic](https://en.wikipedia.org/wiki/Fiber_optic) telecommunication because of low attenuation losses in the SiO2 glass ([silica](https://en.wikipedia.org/wiki/Silica)) medium. [Image intensifiers](https://en.wikipedia.org/wiki/Image_intensifier) are sensitive to this area of the spectrum; examples include [night vision](https://en.wikipedia.org/wiki/Night_vision) devices such as night vision goggles. [Near-infrared spectroscopy](https://en.wikipedia.org/wiki/Near-infrared_spectroscopy) is another common application. |
|                 Short-wavelength infrared                  |                       SWIR, IR-B *DIN*                       |                    1.4–3 μm                     |                         100–214 THz                          |                    413–886 meV                    | 2,070–966 [K](https://en.wikipedia.org/wiki/Kelvin) (1,797–693 [°C](https://en.wikipedia.org/wiki/Celsius)) | Water absorption increases significantly at 1450 nm. The 1530 to 1560 nm range is the dominant spectral region for long-distance telecommunications. |
|                  Mid-wavelength infrared                   | MWIR, IR-C *DIN*; MidIR.[[15\]](https://en.wikipedia.org/wiki/Infrared#cite_note-rdmag20120908-16) Also called intermediate infrared (IIR) |                     3–8 μm                      |                          37–100 THz                          |                    155–413 meV                    | 966–362 [K](https://en.wikipedia.org/wiki/Kelvin) (693–89 [°C](https://en.wikipedia.org/wiki/Celsius)) | In guided missile technology the 3–5 μm portion of this band is the atmospheric window in which the homing heads of passive IR 'heat seeking' missiles are designed to work, homing on to the [Infrared signature](https://en.wikipedia.org/wiki/Infrared_signature) of the target aircraft, typically the jet engine exhaust plume. This region is also known as thermal infrared. |
|                  Long-wavelength infrared                  |                       LWIR, IR-C *DIN*                       |                     8–15 μm                     |                          20–37 THz                           |                    83–155 meV                     | 362–193 [K](https://en.wikipedia.org/wiki/Kelvin) (89 – −80 [°C](https://en.wikipedia.org/wiki/Celsius)) | The "thermal imaging" region, in which sensors can obtain a completely passive image of objects only slightly higher in temperature than room temperature - for example, the human body - based on thermal emissions only and requiring no illumination such as the sun, moon, or infrared illuminator. This region is also called the "thermal infrared". |
| [Far infrared](https://en.wikipedia.org/wiki/Far_infrared) |                             FIR                              |                   15–1000 μm                    |                          0.3–20 THz                          |                    1.2–83 meV                     | 193–3 [K](https://en.wikipedia.org/wiki/Kelvin) (−80.15 – −270.15 [°C](https://en.wikipedia.org/wiki/Celsius)) | (see also [far-infrared laser](https://en.wikipedia.org/wiki/Far-infrared_laser) and [far infrared](https://en.wikipedia.org/wiki/Far_infrared)) |



## Thermal

### Dataset

1. [FREE FLIR Thermal Dataset for Algorithm Training](https://www.flir.com/oem/adas/adas-dataset-form/)

   ![image-20200716161001770](image-20200716161001770.png)

   ![image-20200716161013251](image-20200716161013251.png)


2. [KAIST Multispectral Pedestrian Detection Benchmark](https://soonminhwang.github.io/rgbt-ped-detection/) [2018] [Paper](https://www-users.cs.umn.edu/~jsyoon/JaeShin_homepage/kaist_multispectral.pdf)

   Contain day and night scenarios. Human with bounding box. RGB-Thermal pair.

   The KAIST Multispectral Pedestrian Dataset consists of 95k color-thermal pairs (640x480, 20Hz) taken from a vehicle. All the pairs are manually annotated (person, people, cyclist) for the total of 103,128 dense annotations and 1,182 unique pedestrians. 

   ![teaserImage](teaser.png)

## Real-Multispectral

1. [Hyperspectral Images Database](https://sites.google.com/site/hyperspectralcolorimaging/dataset) [2017]

   **Visible Range MSI**

   NUS hyperspectral images database: 52 Outdoor Scene, 35 Indoor Scene, 33 Individual Fruit Scene, 11 Group Fruit Scene, 13 Real vs Fake Fruit Scene, 44 color Charts & Patches Scene.

  It consists of various indoor and outdoor scenes taken with a SPECIM hyperspectral camera and multiple consumer cameras. For consumer cameras, camera-specific RAW format that is free of any manipulation, is available. For easier classification, this hyperspectral camera dataset has been categorized into the following categories:

  - [General Scenes (Outdoor & Indoor)](https://sites.google.com/site/hyperspectralcolorimaging/dataset/general-scenes)
  - [Fruits](https://sites.google.com/site/hyperspectralcolorimaging/dataset/fruits)
  - [Color Charts and Patches](https://sites.google.com/site/hyperspectralcolorimaging/dataset/color-patches)

  Additionally, our spectral data can be visualized using the professional software by [Scyllarus Matlab/C++ toolbox](http://scyllarus.research.nicta.com.au/).

  Relevant Code [GitHub](https://github.com/trangreyle/gene-color-mapping)

  <img src="image-20200729094343425.png" alt="image-20200729094343425" style="zoom: 50%;" />

  

  <img src="image-20200729094355733.png" alt="image-20200729094355733" style="zoom:50%;" />

  

  <img src="image-20200729094407105.png" alt="image-20200729094407105" style="zoom:50%;" />

  <img src="image-20200729094457881.png" alt="image-20200729094457881" style="zoom:50%;" />

2. [Multispectral Dataset from west virginia university](https://biic.wvu.edu/data-sets/multispectral-dataset)

   1. SWIR Biometrics Dataset: **SWIR**
   2. WVU Multispectral Face Database: Three types of camera are used: **RGB, Multi(RGB+NIR), SWIR**
   3. Multispectral Imaging (Iris) Database: 

3. [Multispectral Image Recognition](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)

   1. Multi-spectral Object Detection

   **RGB, Near-infrared (NIR), Mid-wavelength infrared (MIR), and Far infrared (FIR)** from the left. Objects are labeled and bounding box predicted.

   ![img](det_result.png)

   2. Multi-spectral Semantic Segmentation

   RGB-Thermal dataset with semantic segmentation

   ![img](predictionExamples_good.png)

4. [Multispectral Imaging (MSI) datasets](https://projects.ics.forth.gr/cvrl/msi/): Painting multispectral images. Not paired. Not ordinary objects.

   ![image-20200727175214182](image-20200727175214182.png)

5. [CAVE Multispectral Image Database](https://www.cs.columbia.edu/CAVE/databases/multispectral/)

   **Visible Range MSI:** **400nm to 700nm**

   It only has 32 multispectral & RGB image pairs... Be careful to use it. Each image has 31 bands, and they are separated.

   | Camera              | [Cooled CCD camera (Apogee Alta U260)](http://www.ccd.com/alta_u260.html) |
   | ------------------- | ------------------------------------------------------------ |
   | Resolution          | 512 x 512 pixel                                              |
   | Filter              | [VariSpec liquid crystal tunable filter](http://www.cri-inc.com/products/varispec.asp) |
   | Illuminant          | CIE Standard Illuminant D65                                  |
   | Range of wevelength | 400nm - 700nm                                                |
   | Steps               | 10nm                                                         |
   | Number of band      | 31 band                                                      |
   | Focal length        | f/1.4                                                        |
   | Focus               | Fixed (focused using 550nm image)                            |
   | Image format        | PNG (16bit)                                                  |

   ![img](teaser-20200727174750119.png)

6. [Bristol Hyperspectral Images Database](http://www.cvc.uab.es/color_calibration/Bristol_Hyper/) [1995]

   **Visible Range MSI**

   The database consists of **29 scenes**, each composed by **31 spectrally filtered images** (256 x 256 x 256 grey levels). Each scene has been compressed (zipped) and can be downloaded separately by clicking on the corresponding picture. Please bear in mind that all individual images have a 32 bytes header. To download the whole database at once, just click [here](http://www.cvc.uab.es/color_calibration/Bristol_Hyper/brelstaff.tar.gz).

   There is some code and miscellaneous files [here](http://www.cvc.uab.es/color_calibration/Bristol_Hyper/src/Src.zip) (these need to be run in order to make use of the images as physical measurements). A more complete description on how the images were gathered and some issues on the camera's technicalities can be found [here](http://www.cvc.uab.es/color_calibration/Bristol_Hyper/2-TECH.pdf).

   <img src="image-20200729095320320.png" alt="image-20200729095320320" style="zoom:50%;" />

7. [Harvard Real-World Hyperspectral Images](http://vision.seas.harvard.edu/hyperspec/) [2011]

   **Visible Range MSI:** **420nm to 720nm**

   The camera uses an integrated liquid crystal tunable filter and is capable of acquiring a hyperspectral image by sequentially tuning the filter through a series of **31 narrow wavelength bands**, each with approximately 10nm bandwidth and centered at steps of 10nm from **420nm to 720nm**.

   The captured dataset includes images of both indoor and outdoor scenes featuring a diversity of objects, materials and scale.

   This is a database of **50** hyperspectral images of indoor and outdoor scenes under daylight illumination, and an additional **25** images under artificial and mixed illumination. The images were captured using a commercial hyperspectral camera (Nuance FX, CRI Inc) with an integrated liquid crystal tunable filter capable of acquiring a hyperspectral image by sequentially tuning the filter through a series of thirty-one narrow wavelength bands, each with approximately 10nm bandwidth and centered at steps of 10nm from 420nm to 720nm. The camera is equipped with an apo-chromatic lens and the images were captured with the smallest viable aperture setting, thus largely avoiding chromatic aberration. All the images are of static scenes, with labels to mask out regions with movement during exposure.

   This database is available for non-commercial research use. The data is available as a series of MATLAB .mat files (one for each image) containing both the images data and masks. Since the size of the download is large (around 5.5 + 2.2 GB), we ask that you send an e-mail to the authors at **ayanc[at]eecs[dot]harvard[dot]edu** for the download link. If you use this data in an academic publication, kindly cite the following paper:

   <img src="image-20200729100019504.png" alt="image-20200729100019504" style="zoom:50%;" />

8. [UAE multispectral image database](http://colour.cmp.uea.ac.uk/datasets/multispectral.html)

   **Visible Range MSI:** **400nm to 700nm**

   Wavelength range from 400nm to 700nm at 10nm steps (31 samples). The image matrix for each object is 31xWIDTHxHEIGHT. The images have been captured in a VeriVide viewing booth with a black cloth background under CIE illuminant D75. Each image has been captured twice: once with a white tile and once without. The illuminant has been estimated from the white tile and the spectral data divided by this estimate, in order to arrive at reflectance measurements. The images below are displayed sRGB values rendered under a neutral daylight (D65).

   <img src="image-20200729100842353.png" alt="image-20200729100842353" style="zoom:33%;" />

9. [Manchester hyperspectral images Dataset**s**](http://personalpages.manchester.ac.uk/staff/david.foster/default.html)

   **Visible Range MSI:**  400, 410, ..., 720 nm

   Multiple MSI datasets included:

   <img src="image-20200729101027937.png" alt="image-20200729101027937" style="zoom:50%;" />

   - [Time-Lapse Hyperspectral Radiance Images of Natural Scenes 2015](https://personalpages.manchester.ac.uk/staff/david.foster/Time-Lapse_HSIs/Time-Lapse_HSIs_2015.html)

     <img src="image-20200729101135099.png" alt="image-20200729101135099" style="zoom:33%;" />

   - [Hyperspectral Images for Local Illumination in Natural Scenes 2015](https://personalpages.manchester.ac.uk/staff/david.foster/Local_Illumination_HSIs/Local_Illumination_HSIs_2015.html)

     <img src="image-20200729101232518.png" alt="image-20200729101232518" style="zoom:50%;" />

   - [Hyperspectral Images of Natural Scenes 2002](https://personalpages.manchester.ac.uk/staff/david.foster/Hyperspectral_images_of_natural_scenes_02.html)

     <img src="image-20200729101343922.png" alt="image-20200729101343922" style="zoom:50%;" />

   - [Hyperspectral Images of Natural Scenes 2004](https://personalpages.manchester.ac.uk/staff/david.foster/Hyperspectral_images_of_natural_scenes_04.html)

     <img src="image-20200729101441578.png" alt="image-20200729101441578" style="zoom:50%;" />

10. [BOB NIR+VIS Face Database](https://pythonhosted.org/bob.db.cbsr_nir_vis_2/) [2013]

    It consists of 725 subjects in total. There are [1-22] VIS and [5-50] NIR face images per subject. The eyes positions are also distributed with the images.

    <img src="database.png" alt="_images/database.png" style="zoom:33%;" />

11. [ICVL hyperspectral database](http://icvl.cs.bgu.ac.il/hyperspectral/)

   **RGB+NIR Range MSI:**  Images were collected at 1392×1300 spatial resolution over 519 spectral bands (**400-1,000nm** at roughly 1.25nm increments)

   The database images were acquired using a Specim PS Kappa DX4 hyperspectral camera and a rotary stage for spatial scanning. At this time it contains 201 images and will continue to grow progressively. For your convenience, **.mat** files are provided, downsampled to 31 spectral channels from 400nm to 700nm at 10nm increments.

   <img src="image-20200729102126434.png" alt="image-20200729102126434" style="zoom:50%;" />

11. [University of Granada hyperspectral image database](http://colorimaginglab.ugr.es/pages/Data)

    **RGB+NIR Range MSI:** Most of the images have spatial resolution of 1000 × 900 pixels. The spectral range is from **400 nm to 1000** nm in 10 nm intervals, resulting in total 61 channels.

    <img src="image-20200729102332515.png" alt="image-20200729102332515" style="zoom:50%;" />

12. [SWIRPowder](http://www.cs.cmu.edu/~ILIM/projects/IM/MSPowder/): A 400-1700nm Multispectral Dataset with 100 Powders on Complex Backgrounds

    **SWIR(Multi)+RGB+NIR**

    ![img](result.png)

    ![img](illustration_1.png)

13. [TokyoTech 31-band Hyperspectral Image Dataset](http://www.ok.sc.e.titech.ac.jp/res/MSI/MSIdata31.html) [2015]

    **Visible Range MSI:** **420nm to 720nm**

    Colorful objects with rich textures 30 scenes from 420nm to 720nm at 10nm intervals

    ![MSimage](MSimage.png)

    ![image-20200729094232057](image-20200729094232057.png)



## Reference

- [Spectral Filter Arrays Technology](https://jbthomas.org/TechReport/CIC-shortcourseSFA-2017.pdf)
- [Infrared WiKi](https://en.wikipedia.org/wiki/Infrared)



## Electromagnetic Wave Classification

 γ = [伽马射线](https://zh.wikipedia.org/wiki/伽馬射線)
**[X射线](https://zh.wikipedia.org/wiki/X射線)：**
HX = 硬[X射线](https://zh.wikipedia.org/wiki/X射線)
SX = 软X射线
**[紫外线](https://zh.wikipedia.org/wiki/紫外線)：**
EUV = 极端[紫外线](https://zh.wikipedia.org/wiki/紫外線)
NUV = 近紫外线
**[红外线](https://zh.wikipedia.org/wiki/紅外線)：**
NIR = 近[红外线](https://zh.wikipedia.org/wiki/紅外線)
MIR =中红外线
FIR = [远红外线](https://zh.wikipedia.org/wiki/遠紅外線)

Typically we define near infrared (*NIR*) from 780 nm to 1400 nm and shortwave infrared (*SWIR*) from 1400 nm to 3000 nm.

**[微波](https://zh.wikipedia.org/wiki/微波)：**
EHF = [极高频](https://zh.wikipedia.org/wiki/極高頻)
SHF = [超高频](https://zh.wikipedia.org/wiki/超高頻)
UHF = [特高频](https://zh.wikipedia.org/wiki/特高頻)
**[无线电波](https://zh.wikipedia.org/wiki/無線電波)：**
VHF = [甚高频](https://zh.wikipedia.org/wiki/甚高頻)
HF = [高频](https://zh.wikipedia.org/wiki/高頻)
MF = [中频](https://zh.wikipedia.org/wiki/中頻)
LF = [低频](https://zh.wikipedia.org/wiki/低頻)
VLF = [甚低频](https://zh.wikipedia.org/wiki/甚低频)
ULF = [特低频](https://zh.wikipedia.org/wiki/特低頻)
ELF = [极低频](https://zh.wikipedia.org/wiki/極低頻)