library(igraph)
## please specify root directory for reactome data
dn.rt = 'dn.rt'
## please specify root directory for tcga genetic data
dn.tcga = 'dn.tcga'
rerun.rt = F
if (rerun.rt) {
    library('biomaRt')
    ## Ensembl2Reactome_PE_Pathway.txt is downloaded from Reactome website
    rt = read.table(sprintf('%s/Ensembl2Reactome_PE_Pathway.txt', dn.rt), header=F, sep="\t", quote="", comment.char="")
    colnames(rt) = c('src_db_id', 'rt_pe_sid', 'rt_pe_nm', 'rt_pathway_id', 'url','event_nm', 'evid_code', 'species')

    rt.hs = rt[rt$species == 'Homo sapiens',]

    mart <- useDataset("hsapiens_gene_ensembl", useMart("ensembl"))
    genes <- rt.hs$src_db_id
    G_list <- unique(getBM(filters= "ensembl_gene_id", attributes= c("ensembl_gene_id", "hgnc_symbol"), values=genes, mart= mart))
    rt.hs.gene = merge(rt.hs,G_list,by.x="src_db_id",by.y="ensembl_gene_id")
    save(rt.hs.gene, file=sprintf('%s/Gene2Reactome_PE_Pathway.RData', dn.rt))
}else {
    load(sprintf('%s/Gene2Reactome_PE_Pathway.RData', dn.rt))
}
g2rt = unique(rt.hs.gene[,c('hgnc_symbol', 'rt_pathway_id')])

tcga.raw = read.table(sprintf('%s/tcga_raw_merged.spmat', dn.tcga), header=T, sep='\t', quote="")
## generate sparse tensor with label information, columns are:
## gene, subject id, cancer type, variant count, reactome pathway id
tcga.pathway = merge(tcga.raw, g2rt, by.x="gene", by.y="hgnc_symbol")
write.csv(tcga.pathway, file=sprintf('%s/tcga_pathway.spt', dn.tcga, mode, sel), row.names=F, quote=F)

ces = unique(tcga.pathway[,c('hgnc_symbol', 'rt_pathway_id')])
write.csv(ces, file=sprintf('%s/rt_pathway.csv', dn.tcga, mode, sel), row.names=F, quote=F)
g = graph.data.frame(ces, directed=FALSE)
vertexAttrs = vertex.attributes(g)
clusterRes = clusters(g) 
eclusters = list()
for (i in 1:clusterRes$no) {
    eclusters[[i]] = vector()
}
for (j in 1:length(clusterRes$membership)) {
    cid = clusterRes$membership[j] # cluster id
    eid = length(eclusters[[cid]]) + 1 # element id
    eclusters[[cid]][eid] = vertexAttrs$name[j]
}
