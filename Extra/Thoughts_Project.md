
As for segmenting the healty tissue as well (for example the lymphnodes) -> not right now because of time, but could be an extention if we have time, we would change it to a two decoder network, one for segmenting the tumor and then one for segmenting the healty tissue/lymphnodes, that way you know there should not be overlap -> might help when segmenting the problem to make sure the model does not classify the lymphnodes as (part of the) tumor. -> Nils did not do this, but at least could be for future research. OG dataset does have lymphnode ground truth segmentations, the extended (except for the 10 mr_subset), does not have these other ground-truth segmentations, so it cannot be used for that at least. 



Two extra datasets that can still be used (outside of the original) are:
- prediction_intratreatment: no ground truths/labels?? but if there are, it's not clear which files are labels and to what scan they belong to? -> Maybe ask Roland and look with ITK-snap (remember that this could be very similar to OG dataset -> look at data)
- OPSCC_VUMC also is in DCM format but doesnt seem to have the corresponding MRI scan

Other datasets are disregarded because:
KNO_retrospective (DWI data - thomas) not using because T1 and ADC only
AvL_Radiomics_study_T1g_ADC_DWWI - not using this dataset because of high variability


At first we said: KNO_retrospective (DWI data - thomas) not using because T1 and ADC only, but since the extra avl datatset also exists out of many 2 scan instances we might still be able to use this dataset as well. It might be a good idea to talk about it with Yunchao unless we know we don't have time to do this anyways. But beforre we talk about this with Yunchao, please look at the files to see if it is usable. Is the data similar enough to use it for our experiment, can we open the data points, are the labels included and can we open them etc.? We should also look if the dataset does not have too much variability just like the first option we had. Otherwise this could alter our preprocessing steps even more.
This is another point to think about, if we are going to use the extra data, should we preprocess them more than the standard nnunet preprocessing (z-score). Because our current idea is that we do not use the other methods because they don't seem to be better (this is in the nnunet paper or github -> look it up), and becuase Nils had already tried other things on our original datatset and that didn't work better. But for the extra data, this second argument doesn't exist. I mean we could argue for consistency to use the same preprocessing steps, but it is pretty common to base your preprocessing steps on your dataset... We could also argue that the extra data is (at least partially) very similar to our original dataset which means that it is likely that the same preprocessingsteps that work for the original dataset also work well for the extra data. It might also be a good idea to look into the seq2seq paper to see what they use for preprocessing steps and then we could say see they also use these preprocessing steps (if they also use the nnunet preprocessing pipeline), so it is good to use these techniques for generating the missing MRI sequences as well.


Please check if labels are actually only 0 and 1s (Nils had code to convert the labels to binary because "they weren't entirely binary", not sure which data this is about)


while registering the extra OG data, we register 0003 on 0000 unless for one patient where 0000 is missing and we register on 0005 (as that was the clearest scan avaiable)

Flipped data:
083_0000 and 083_0003 (OG extra data)
"092_0000", "092_0002", "092_0003", "093_0000", "097_0003", "100_003", "106_0002",
    "107_0002", "108_0000", "108_0003", "109_0000", "109_0003", "110_0000", "110_0003",
    "113_0002", "115_0000", "115_0003", "116_0000", "116_0003", "118_0002", "119_0002",
    "120_0000", "121_0000", "121_0003", "123_0000", "123_0003" (avl extra data)

OG:
14, 64 and 69 (all files after nn)


The seq2seq model actually does not work for 3D, so we'll use the 2D version

About OPSCC_VUMC:
We should have:
        "0": "T1",
        "1": "T1C",
        "2": "T2",
        "3": "STIR",
        "4": "DWI",
        "5": "ADC"
But we have:
        T2, STIR, T1, DCE
        But sometimes:
        No T2
        and/or:
        No DCE

Around 31 images with at least 3 sequences, but some of them have DCE instead of T2 so then we still need to generate 4 sequences... maybe also filter out the ones without T2?


3 STIR T1 DWI T1C
4 STIR T1 DWI ADC T1C
9 STIR T1 DWI ADC T1C
11 STIR T1 DWI T1C
67 STIR T1 ADC T1C T2
102 STIR T1 DWI T1C T2
131 STIR T1 DWI ADC T2 
143 STIR T1 DWI T1C T2
146 STIR DWI ADC
172 STIR T1 DWI ADC T1C
173 STIR T1 DWI ADC T2


which scans are missing from the extra data --> make table for in the apendix of which scans are generated for all the extra data and refer to that table the respective part of the dataset
extra original in order of map 78-88:
1
2
25
2
2
25
4
5
1
5
012




train 

['000', '006', '007', '008', '013', '014', '015', '016', '017', '018', '019', '020', '022', '023', '024', '025', '026', '029', '031', '032', '033', '034', '035', '037', '038', '039', '040', '041', '042', '043', '045', '047', '048', '049', '050', '052', '053', '055', '056', '058', '059', '062', '063', '068', '069', '070', '071', '073', '074', '075', '077', '078', '081', '083', '084', '085', '088', '089', '091', '093', '094', '096', '097', '098', '099', '124', '130', '141', '155', '160', '165', '170', '179', '188', '195']


000 x 001
001 x 005
002 x 021
003 x 027
004 x 028
005 x 030
006 x 036
007 x 044
008 x 065
009 x 066
010 x 076
011 x 081
012 x 092
013 x 095
014 x 120
015 x 137




- all six real: original results
- all six generated: 541, "allGen"
- only all T1 generated: 542, "T1Gen"
- only all T1C generated: 543, "T1CGen"
- only all T2 generated: 544, "T2Gen"
- only all STIR generated: 545, "STIRGen"
- only all DWI generated: 546, "DWIGen"
- only all ADC generated: 547, "ADCGen"
- only all DWI and ADC generated: 548, "DWI_ADC_Gen"
- 30 % generated T2 and other 30% generated STIR: 551, "30T2_30STIR"
- 30 % generated T2 and other 30% generated STIR +  all DWI and ADC generated: 552, "30T2_30STIR_ALLDWIADC"
- 30 % generated T2 and other 30% generated STIR +  60% both DWI and ADC generated: 553, "30T2_30STIR_60DWIADC"
- 30 % generated T1 and other 30% generated T1C: 554, "30T1_30T1C"
- 30 % generated T1 and other 30% generated T1C +  60% both DWI and ADC generated: 555, "30T1_30T1C_60DWIADC"
- 30 % generated T1 and other 30% generated T1C +  60% both DWI and ADC generated + 30 % generated T2 and other 30% generated STIR: 556, "30T1_30T1C_60DWIADC_30T2_30STIR"



TP
(True Positive)	Generated image correctly guessed as generated
FP
(False Positive)	Real image wrongly guessed as generated
FN
(False Negative)	Generated image wrongly guessed as real
TN
(True Negative)	Real image correctly guessed as real


Gen:
2 T1
5 STIR
6 DWI
9 T1C
10 T2
13 ADC
14 T1
16 T2
19 ADC

Real:
3 T1C
4 T2
7 ADC
8 T1
11 STIR
12 DWI
15 T1C
17 STIR
18 DWI

guesses:
2 gen
3 real
4 real
5 gen
6 gen
7 real
8 real
9 gen
10 gen
11 gen
12 real
13 gen
14 gen
15 real
16 real
17 gen
18 real
19 gen














000 Guess: Red Prediction
Green R-GT1
Blue S-GT2
Red Prediction

001 Guess: Blue R-GT1
Green Prediction
Blue R-GT1
Red S-GT2

002 Guess: Blue R-GT1
Green S-GT2
Blue R-GT1
Red Prediction

003 Guess: Red S-GT2
Green R-GT1
Blue Prediction
Red S-GT2

004 Guess: Red Prediction
Green S-GT2
Blue R-GT1
Red Prediction

005 Guess: Red S-GT2
Green R-GT1
Blue Prediction
Red S-GT2

006 Guess: Blue Prediction
Green R-GT1
Blue Prediction
Red S-GT2

007 Guess: Blue R-GT1
Green S-GT2
Blue R-GT1
Red Prediction

008 Guess: Green Prediction
Green Prediction
Blue S-GT2
Red R-GT1

009 Guess: Blue R-GT1
Green S-GT2
Blue R-GT1
Red Prediction

010 Guess: Blue R-GT1
Green Prediction
Blue R-GT1
Red S-GT2

011 Guess: Red R-GT1
Green S-GT2
Blue Prediction
Red R-GT1

012 Guess: Blue Prediction
Green R-GT1
Blue Prediction
Red S-GT2

013 Guess: Blue R-GT1
Green Prediction
Blue R-GT1
Red S-GT2

014 Guess: Green S-GT2
Green S-GT2
Blue R-GT1
Red Prediction

015 Guess: Blue S-GT2
Green R-GT1
Blue S-GT2
Red Prediction

