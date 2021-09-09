## Predicting COVID-19 Pneumonia Severity on Chest X-ray with Deep Learning

🛑 NOT FOR MEDICAL USE 🛑 

There are now two datasets:

1. Scored severity for the COVID-19 Image Data Collection Dataset is here: [Pneumonia severity scores for 94 images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/annotations/covid-severity-scores.csv)
2. Stonybrook Radiographic Assessment of Lung Opacity (RALO) dataset is here: [Pneumonia severity scores for 2373 images](https://zenodo.org/record/4634000)

The scores are explained as follows for the RALO dataset:
  - **geographic_extent_mean**: The extent of lung involvement by ground glass opacity or consolidation for each lung (right lung and left lung separately) was scored as: 0 = no involvement; 1 = <25% involvement; 2 = 25-50% involvement; 3 = 50-75% involvement; 4 = >75% involvement. The total extent score ranged from 0 to 8 (right lung and left lung together). 
  - **opacity_mean**: The degree of opacity for each lung (right lung and left lung separately) was scored as: 0 = no opacity; 1 = ground glass opacity; 2 = consolidation; 3 = mix of consolidation and ground glass opacity (>50% consolidation); 4 = white-out. The total opacity score ranged from 0 to 8 (right lung and left lung together). NOTE: The total opacity score ranged from 0 to 6 for the COVID-19 Image Data Collection Dataset so scalling (like opacity/6\*8) will align the two datasets.


Data is here: [Pneumonia severity scores for 94 images](https://github.com/ieee8023/covid-chestxray-dataset/blob/master/annotations/covid-severity-scores.csv)



License: CC BY-SA Creative Commons Attribution-ShareAlike

These are from the follow paper:
Cohen, Joseph Paul, et al. Predicting COVID-19 Pneumonia Severity on Chest X-Ray with Deep Learning. Cureus Medical Journal, 10.7759/cureus.9448, http://arxiv.org/abs/2005.11856.

```bibtex
@article{Cohen2020Severity,
title = {Predicting COVID-19 Pneumonia Severity on Chest X-ray with Deep Learning},
author = {Cohen, Joseph Paul and Dao, Lan and Morrison, Paul and Roth, Karsten and Bengio, Yoshua and Shen, Beiyi and Abbasi, Almas and Hoshmand-Kochi, Mahsa and Ghassemi, Marzyeh and Li, Haifang and Duong, Tim Q},
journal = {Cureus Medical Journal},
doi = {10.7759/cureus.9448}
url = {https://www.cureus.com/articles/35692-predicting-covid-19-pneumonia-severity-on-chest-x-ray-with-deep-learning},
year = {2020}
}
```

To run the CLI:

```bash
# basic command line predictions
$ python predict_severity.py 2966893D-5DDF-4B68-9E2B-4979D5956C8E.jpeg
geographic_extent (0-8): 5.978744940174467
opacity (0-6): 4.169582852893416

# or to output a saliency map:
$ python predict_severity.py 01E392EE-69F9-4E33-BFCE-E5C968654078.jpeg  -saliency_path heatmap.jpg

```

| Image             | Saliency map | Predictions       |
|-------------------------------|-------------------------------|-----------|
| ![](examples/2966893D-5DDF-4B68-9E2B-4979D5956C8E.jpeg-resize.jpg)| ![](examples/2966893D-5DDF-4B68-9E2B-4979D5956C8E.jpeg-heatmap.jpg)| geographic_extent (0-8): 5.979 <br>opacity (0-6): 4.17 |
| ![](examples/31BA3780-2323-493F-8AED-62081B9C383B.jpeg-resize.jpg)| ![](examples/31BA3780-2323-493F-8AED-62081B9C383B.jpeg-heatmap.jpg)| geographic_extent (0-8): 6.293 <br>opacity (0-6): 4.367 |
| ![](examples/41591_2020_819_Fig1_HTML.webp-day5.png-resize.jpg)| ![](examples/41591_2020_819_Fig1_HTML.webp-day5.png-heatmap.jpg)| geographic_extent (0-8): 3.067 <br>opacity (0-6): 2.335 |
| ![](examples/8FDE8DBA-CFBD-4B4C-B1A4-6F36A93B7E87.jpeg-resize.jpg)| ![](examples/8FDE8DBA-CFBD-4B4C-B1A4-6F36A93B7E87.jpeg-heatmap.jpg)| geographic_extent (0-8): 0.9483 <br>opacity (0-6): 1.0 |

