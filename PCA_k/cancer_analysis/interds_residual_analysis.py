import pandas as pd

ds1_residuals_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/residuals.csv'
residual_df1 = pd.read_csv(ds1_residuals_csv_fp)

ds2_residuals_csv_fp = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/residuals.csv'
residual_df2 = pd.read_csv(ds2_residuals_csv_fp)

#plot whisker plot of median residuals for cases with 0 cancer or cyst, and those with cancer or cyst
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# change matplotlib backend to interactive display on linux pycharm
plt.switch_backend('TkAgg')

threshold = 10000

residual_df1['name'] = np.where(((residual_df1['largest_cancer'] <threshold) & (residual_df1['largest_cyst'] < threshold)) > 0, 'normal', 'abnormal')
residual_df2['name'] = np.where(((residual_df2['largest_cancer'] <threshold) & (residual_df2['largest_cyst'] < threshold)) > 0, 'normal', 'abnormal')

#drop cases with no cysts or cancers in df1
residual_df1 = residual_df1.loc[residual_df1['name'] == 'abnormal']
#drop cases with cysts or cancers in df2
residual_df2 = residual_df2.loc[residual_df2['name'] == 'normal']

#merge with df2
residual_df = pd.concat([residual_df1,residual_df2])


#calculate p values for median_distance, mean_distance, max_distance, and 90th_percentile_distance between normal and abnormal cases
# we want to correct for multiple tests, so we use the bonferroni correction
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

p_values = []
for col in ['median_distance','mean_distance','max_distance','90th_percentile_distance']:
    p = mannwhitneyu(residual_df.loc[residual_df['name'] == 'normal'][col],
                                 residual_df.loc[residual_df['name'] == 'abnormal'][col])[1]
    p_values.append(p)

#correct for 4 tests and 2 thresholds - 8 tests total
p_values = multipletests(p_values*2,method='bonferroni')[1]

# plot 2x2 grid of boxplots to show the distribution of median_distance, mean_distance, max_distance, and 90th_percentile_distance
# for normal and abnormal cases
fig, axes = plt.subplots(2,2,figsize=(10,10))
sns.boxplot(ax=axes[0,0],x='name',y='median_distance',data=residual_df)
axes[0,0].set(xlabel='Kidney Type', ylabel='median_distance')
#add p value to plot
axes[0,0].text(0.5,0.9,'p = {:.3e}'.format(p_values[0]),horizontalalignment='center',verticalalignment='center',transform=axes[0,0].transAxes)

sns.boxplot(ax=axes[0,1],x='name',y='mean_distance',data=residual_df)
axes[0,1].set(xlabel='Kidney Type', ylabel='mean_distance')
axes[0,1].text(0.5,0.9,'p = {:.3e}'.format(p_values[1]),horizontalalignment='center',verticalalignment='center',transform=axes[0,1].transAxes)

sns.boxplot(ax=axes[1,0],x='name',y='max_distance',data=residual_df)
axes[1,0].set(xlabel='Kidney Type', ylabel='max_distance')
axes[1,0].text(0.5,0.9,'p = {:.3e}'.format(p_values[2]),horizontalalignment='center',verticalalignment='center',transform=axes[1,0].transAxes)

sns.boxplot(ax=axes[1,1],x='name',y='90th_percentile_distance',data=residual_df)
axes[1,1].set(xlabel='Kidney Type', ylabel='90th_percentile_distance')
axes[1,1].text(0.5,0.9,'p = {:.3e}'.format(p_values[3]),horizontalalignment='center',verticalalignment='center',transform=axes[1,1].transAxes)
#sns save figure
plt.savefig('/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/residuals_boxplot.png')
plt.show()
