import pandas as pd
from scipy.sparse import coo_matrix
import pickle

def tensor_label_encoding(spt,
                          fntout='tcga_spt_num.csv',
                          fnfrac='tcga_spt_frac.csv',
                          fngt='tcga_gt.csv',
                          fnglab='genes.csv',
                          fnilab='subj_ids.csv',
                          fnplab='rt_pathway_ids.csv',
                          ddn=''):
    glabels, guniques = spt['gene'].factorize(sort=True)
    pd.DataFrame(guniques).to_csv('%s/%s' % (ddn, fnglab), header=False, index=False)

    ilabels, iuniques = spt['id'].factorize(sort=True)
    pd.DataFrame(iuniques).to_csv('%s/%s' % (ddn, fnilab), header=False, index=False)

    plabels, puniques = spt['rt_pathway_id'].factorize(sort=True)
    pd.DataFrame(puniques).to_csv('%s/%s' % (ddn, fnplab), header=False, index=False)

    spt_num = spt.copy()
    spt_num['id'] = ilabels
    spt_num['gene'] = glabels
    spt_num['rt_pathway_id'] = plabels
    spt_num = spt_num[['id', 'rt_pathway_id', 'gene', 'va_cnt']] 
    spt_num.to_csv('%s/%s' % (ddn, fntout), index=False)

    gp = spt[['gene', 'rt_pathway_id']].drop_duplicates()
    gpcnt = gp.groupby('gene').count()
    gpcnt.columns = ['pwcnt']
    npw = len(puniques)
    spt_frac = spt.copy()
    spt_frac['size'] = np.array(gpcnt.loc[spt_frac['gene'],'pwcnt'])
    spt_frac['id'] = ilabels
    spt_frac['gene'] = glabels
    spt_frac['rt_pathway_id'] = plabels
    spt_frac['va_cnt'] = np.log(spt_frac['va_cnt']+1) * np.log(npw/spt_frac['size']) # tfidf
    spt_frac = spt_frac[['id', 'rt_pathway_id', 'gene', 'va_cnt']] 
    spt_frac.to_csv('%s/%s' % (ddn, fnfrac), index=False)
    
    gt = spt[['id', 'ca_type']]
    gt = gt.drop_duplicates()
    gt.sort_values('id').to_csv('%s/%s' % (ddn, fngt), index=False)
    return (spt_num, spt_frac, ilabels, iuniques, plabels, puniques, glabels, guniques)

# TODO: please specify your root genetic data directory
ddn = 'genetic data root'
tcga_spt = pd.read_csv(f'{ddn}/tcga_pathway.spt') 

psz = pd.read_csv(f'{ddn}/rt_pathway_nosubiso.csv') 
tcga_spt = tcga_spt.loc[tcga_spt['rt_pathway_id'].isin(psz['rt_pathway_id'])]

tcga_gene = tcga_spt[['id','gene']].drop_duplicates()
npt = tcga_gene['id'].nunique()
print('#pt: %d' % (npt))
gene_tally = tcga_gene['gene'].value_counts()
genes_gt5p = gene_tally[gene_tally>=npt*.05].index

# ggt5p_sgt10
tcga_ggt5p = tcga_spt[tcga_spt['gene'].isin(genes_gt5p)]
tcga_gene2 = tcga_ggt5p[['id','gene']].drop_duplicates()
subj_tally = tcga_gene2['id'].value_counts()

subj_gt10 = subj_tally[subj_tally>=10].index # doesn't have actual effect
tcga_ggt5p_sgt10 = tcga_ggt5p[tcga_ggt5p['id'].isin(subj_gt10)]

(tcga_spt_num, 
 tcga_spt_frac,
 ilabels, 
 iuniques, 
 plabels, 
 puniques, 
 glabels, 
 guniques) = tensor_label_encoding(tcga_spt_ggt5p_sgt10,
                                   fntout='tcga_spt_num.csv',
                                   fnfrac='tcga_spt_frac.csv', 
                                   fngt='tcga_gt.csv',
                                   fnglab='genes.csv', 
                                   fnilab='subj_ids.csv',
                                   fnplab='rt_pathway_ids.csv', 
                                   ddn=ddn)

tcga_spt_num = pd.read_csv(f'{ddn}/tcga_spt_num.csv')
tcga_gt = pd.read_csv(f'{ddn}/tcga_gt.csv')
tcga_spt_num = tcga_spt_num[['id', 'rt_pathway_id', 'gene', 'va_cnt']]

tcga_spm = tcga_spt_num[['id', 'gene', 'va_cnt']].drop_duplicates()
tcga_spm.to_csv(f'{ddn}/tcga_spm.csv', index=False)

pt_gene = coo_matrix((tcga_spm['va_cnt'], (tcga_spm['id'], tcga_spm['gene'])))
tcga_mat = pd.DataFrame(pt_gene.todense(), columns=guniques, index=iuniques)
y = tcga_gt.sort_values('id')['ca_type'].values

f = open(f'{ddn}/tcga.pik', 'wb')
pickle.dump([tcga_mat,y], f, -1)
f.close()


pathway_spm = tcga_spt_num[['id', 'rt_pathway_id', 'va_cnt']]
pathway_cnt = pathway_spm.groupby(['id', 'rt_pathway_id'])['va_cnt'].sum()
pathway_cnt.to_csv(f'{ddn}/pathway_spm.csv', header=['va_cnt'])

pathway_cnt = pd.read_csv(f'{ddn}/pathway_spm.csv')
pt_pathway = coo_matrix((pathway_cnt['va_cnt'], (pathway_cnt['id'], pathway_cnt['rt_pathway_id'])))
pathway_mat = pd.DataFrame(pt_pathway.todense(), columns=puniques, index=iuniques)
y = tcga_gt.sort_values('id')['ca_type'].values
f = open(f'{ddn}/tcga_pm.pik', 'wb')
pickle.dump([pathway_mat,y], f, -1)
f.close()

npt = tcga_spt_num['id'].max()+1
npw = tcga_spt_num['rt_pathway_id'].max()+1
nge = tcga_spt_num['gene'].max()+1
print(npt, npw, nge)
tcga_t = np.zeros((npt, npw, nge), dtype=np.float32)

for index, row in tcga_spt_num.iterrows():
    tcga_t[row['id'], row['rt_pathway_id'], row['gene']] = row['va_cnt']

f = open(f'{ddn}/tcga_t.pik', 'wb')
pickle.dump([tcga_t,y], f, -1)
f.close()


# this is the co-occurrence counting scheme described in the paper. Intuitively, the heuristic counts additional gene-pathway co-occurrence where the patient has some variants that hit a gene and other variants that hit a pathway not containing the gene. This generalized by-patient co-occurrence is motivated by the fact that we currently do not know all possible interactions between genes and pathways, as knowledge is still evolving and knowledge-based co-occurrence can be incomplete.
npt, npw, ng = tcga_t.shape
pcnt = 0
for ipt in range(npt):
    for ipw in range(npw):
        for ig in range(ng):
            if tcga_t[ipt, ipw, ig] == 0:
                if tcga_pmat.iloc[ipt, ipw] !=0 and tcga_mat.iloc[ipt, ig] !=0:
                    if tcga_pmat.iloc[ipt, ipw] >= tcga_mat.iloc[ipt, ig]:
                        tcga_t[ipt, ipw, ig] = tcga_mat.iloc[ipt, ig]
                    else:
                        tcga_t[ipt, ipw, ig] = tcga_pmat.iloc[ipt, ipw]
                        pcnt += 1

print( (tcga_t!=0).sum() / np.prod(tcga_t.shape) )
print( pcnt )

f = open(f'{ddn}/tcga_tco2.pik', 'wb')
pickle.dump([tcga_t,y], f, -1)
f.close()
