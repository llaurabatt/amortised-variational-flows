rm(list = ls())

library(rstan)
library(dplyr)
options(mc.cores=4)

#wd <- '/home/manderso/Documents/StatML 2021-22/MiniProject 1/Spatial Texts Stan/HPC April/'
wd = '/rds/general/user/mma22/home/'
eta = 1
type = 2
n_float = 30
n_items = 8
n_iter = 2000
descr= 'l0.5_Q5'

data_coarsened_all_items = readRDS(paste0(wd,'coarsened_all_items_v2'))

data.loc <- data_coarsened_all_items$data.loc
data.y <- data_coarsened_all_items$data.m
unique_forms = unique(data_coarsened_all_items$data.items)
data.n_forms = c(sapply(unique_forms, function(x)sum(data_coarsened_all_items$data.items==x)))

#actual floating profiles
to.float <- c(1,2,3,4,5,16,17,23,26,29,30,32,36,38,43,45,46,49,51,52,54,55,56,60,61,62,65,68,69,70,71,73,75,79,80,99,100,106,110,114,115,130,136,140,154,164,165,167,168,169,175,177,180,181,183,184,186,188,189,192,193,194,196,198,200,201,202,204,206,207,210,212,213,215,217,219,220,221,222,223,225,226,227,234,235,236,237,238,240,243,246,247,257,259,260,277,278,287,299,300,302,303,314,316,319,320,322,325,357,358,361,365,382,398,405,410,411,419,422,423,425,426,432,434,435,454,461,473,474,476,477,479,488,491,492,494,495,496,497,498,500,501,503,504,505,506,507,508,509,510,511,512,514,515,516,517,519,527,529,530,531,534,536,537,539,540,541,549,550,551,552,553,554,556,557,558,559,560,561,577,578,579,580,581,582,583,584,587,588,591,593,597,603,605,652,661,676,677,678,699,704,709,714,715,717,718,726,729,730,736,737,738,742,752,755,761,763,764,766,767,804,901,905,908,910,912,913,927,1002,1300,1352,4218,4239,4245,4285,4286,4289,4675,4682,4685,7550,7591,7592,7593,7600,7610,7980)

# alternative selection of floating profiles that are better mixed with the anchor profiles
# to.float <- c(1,5,23,29,30,32,38,45,51,54,55,61,65,68,70,71,73,110,114,115,130,136,140,154, 164,165,167,168,169,175,177,180,181,186,188,192,196,201,206,210,212,213,217,220,221,223,225,234, 235,236,237,238,246,257,259,278,287,299,300,302,303,319,325,358,361,365,419,422,423,425,434,435, 479,488,492,498,504,505,506,507,510,511,515,516,517,531,534,537,540,549,554,558,559,561,577,578, 579,580,581,587,603,676,714,717,718,730,738,742,752,767,910,1002,4218,4285,4286,4675,4682,4685,7550,7591,7600,7610,7980)

if (type == 1){
  to.float <- to.float[1:n_float]
}
float_indexes = which(data.loc[,1] %in% to.float)

#generate the inducing points
inducing_points = expand.grid(seq(0,1, length.out=11),  seq(0,0.9, length.out=10))
n_inducing_points = NROW(inducing_points)

#select which items will be used for inference
item_prevalence = c()
for (i in 1:length(data.n_forms)){
  item_prevalence = c(item_prevalence, sum(data.y[(sum(data.n_forms[0:(i-1)])+1):sum(data.n_forms[1:i]),]))
}
#items_used = sort(item_prevalence, index.return=TRUE, decreasing=TRUE)$ix[1:n_items]
# original 8 items = c(26,24,4,23,65,71,20,12)
items_used = c(26,24,4,23,65,71,20,12)
items_used = sort(items_used)

#obtain the indexes of the y data corresponding to these items
y_form_indexes= c()
for (i in items_used){
  y_form_indexes = c(y_form_indexes,(sum(data.n_forms[0:(i-1)])+1):sum(data.n_forms[1:i]))
}

#sort y data into anchor profiles followed by floating profiles
y.sort = cbind(data.y[y_form_indexes,-float_indexes], data.y[y_form_indexes,float_indexes[1:n_float]])

data.loc.float = sweep(data.loc[float_indexes[1:n_float],2:3], 2, c(350,275)) / 200
data.loc.anchor = sweep(data.loc[-float_indexes,2:3],2, c(350,275)) / 200

text_data <- list(n_items = length(data.n_forms[items_used]),
                  n_forms_total = sum(data.n_forms[items_used]),
                  n_forms = data.n_forms[items_used],
                  n_locs = NROW(data.loc.anchor) + NROW(data.loc.float),
                  n_anchor = NROW(data.loc.anchor),
                  n_floating = NROW(data.loc.float),
                  n_inducing = 110,
                  y = t(y.sort),
                  anchor_locs = data.loc.anchor,
                  inducing_locs = inducing_points,
                  gp_magnitude = 1,
                  gp_length_scale = 0.5,
                  Q= 5,
                  eta_smi = eta)

output_name = paste0('fit_smi_', descr, '_iter', n_iter, '_eta', eta, '_items', n_items, '_float', n_float, '_type', type)

model_stan = stan_model(paste0(wd, 'module1.stan'))
smi_fit_part1 = sampling(model_stan, data = text_data, iter=n_iter, chains=3, cores=3, warmup=500)

saveRDS(smi_fit_part1, output_name)
