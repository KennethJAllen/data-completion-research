# graduate_research
My graduate research on low-rank matrix and tensor completion, and maximum volume algorithms for finding dominant submatrices. 

Consider the following 128 by 128 image of a penny.

![128 by 128 image of a penny](https://raw.githubusercontent.com/KennethJAllen/graduate_research/main/Low_Rank_Matrix_Completion/Image_Completion/Image_Examples/penny_picture.jpg)

Suppose 25% of it is deleted at random such as the following.

![25% of penny missing](https://raw.githubusercontent.com/KennethJAllen/graduate_research/main/Low_Rank_Matrix_Completion/Image_Completion/Image_Examples/three_fourths_partially_known_penny.jpg)

We would like to recover the missing entries. Using AlternatingProjection.m in Low_Rank_Matrix_Completion we recover the following image.

![penny recovered with alternating projection](https://raw.githubusercontent.com/KennethJAllen/graduate_research/main/Low_Rank_Matrix_Completion/Image_Completion/Image_Examples/alt_proj_recovered_penny_rank18.jpg)

Using Schur_MV_Grad_Descent.m we recover the following image.

![penny recovered with alternating projection](https://raw.githubusercontent.com/KennethJAllen/graduate_research/main/Low_Rank_Matrix_Completion/Image_Completion/Image_Examples/MVGD_recovered_penny_rank_18.jpg)
