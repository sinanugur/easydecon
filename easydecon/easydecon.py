import scanpy as sc
import numpy as np
import pandas as pd
import spatialdata as sp
import spatialdata_io


def group_gene_expression(sdata, genes,group, bin_size=8,quantile=0.70):
    #available_genes = sdata.tables[f"square_00{bin_size}um"].var_names
    #filtered_genes = [gene for gene in genes if gene in available_genes]
    table_key = f"square_00{bin_size}um"
    table = sdata.tables[table_key]
    filtered_genes = list(set(genes).intersection(table.var_names))
    gene_expression = table[:, filtered_genes].to_df().sum(axis=1).to_frame(group)
    #gene_expression = gene_expression.sum(axis=1).to_frame(group)
    #gene_expression['id'] = gene_expression.index
    if group in table.obs.columns:
        table.obs.drop(columns=[group], inplace=True)
    
    threshold=gene_expression[gene_expression[group] !=0].quantile(quantile)
    gene_expression[group] = np.where(gene_expression[group].values >= threshold.values, gene_expression[group], 0)

    table.obs=pd.merge(table.obs, gene_expression, left_index=True, right_index=True)
    
    
    return gene_expression

def read_cluster_markers(filename,sdata,exclude_clusters=[],bin_size=8,top_n_genes=60): #100
    
    try:
        df=pd.read_csv(filename,dtype={"names":str,"group":str})
    except:
        df=pd.read_excel(filename,dtype={"names":str,"group":str})
    df=df[df["names"].isin(sdata.tables[f"square_00{bin_size}um"].var_names)]
    df=df[~df['group'].isin(exclude_clusters)]
    df = df.sort_values(by='scores', ascending=False)
    df = df.groupby('group').head(top_n_genes)
    df.set_index("group",inplace=True)
    return df

def function_identify_cluster(sdata,cluster_membership_df,group,bin_size=8):
    table = sdata.tables[f"square_00{bin_size}um"]
    associated_cluster=dict()
    spots_with_expression = table.obs[table.obs[group] != 0].index


    
    for spot in spots_with_expression:
        a=dict()

        for cluster in cluster_membership_df.index.unique():
            #genes=cluster_membership_df.loc[cluster]["names"].values
            genes=cluster_membership_df.loc[cluster]["names"]
            if isinstance(genes, str):
                genes = [genes]
            else:
                genes = genes.values
            #group_expression = sdata.tables[f"square_00{bin_size}um"][spot, genes].to_df().sum(axis=1).values
            #group_expression = sdata.tables[f"square_00{bin_size}um"][spot, genes].to_df().apply(gmean,axis=1).values
            group_expression = table[spot, genes].to_df().mean(axis=1).values
            a[cluster]=group_expression
        #max_cluster=str(max(a, key=a.get))

        associated_cluster[spot]=str(max(a, key=a.get))
    
    for spot in set(table.obs.index) - set(spots_with_expression):
        associated_cluster[spot] = None
        
    df=pd.DataFrame(list(associated_cluster.items()), columns=['Index', 'assigned_cluster'])
    df.set_index('Index', inplace=True)
    df[f'{group}_clusters'] = pd.Categorical(df['assigned_cluster'],categories=cluster_membership_df.index.unique())
    df.drop(columns=['assigned_cluster'],inplace=True)
    return df

def visualize_cluster_expression(sdata,cluster_membership_df,group,bin_size=8):
    table = sdata.tables[f"square_00{bin_size}um"]
    spots_with_expression = table.obs[table.obs[group] != 0].index
    
    for spot in spots_with_expression:
        a=dict()

        for cluster in cluster_membership_df.index.unique():
            #genes=cluster_membership_df.loc[cluster]["names"].values
            genes=cluster_membership_df.loc[cluster]["names"]
            if isinstance(genes, str):
                genes = [genes]
            else:
                genes = genes.values
            group_expression = table[spot, genes].to_df().mean(axis=1).values
            a[cluster]=group_expression

        tmp=pd.DataFrame.from_dict(a, orient='index').transpose()
        tmp.index=[spot]
        if 'df' in locals():
            df=pd.concat([df,tmp])
        else:
            df=tmp
    for spot in set(table.obs.index) - set(spots_with_expression):
        df=pd.concat([df,pd.DataFrame(index=[spot],columns=df.columns)])
    df.fillna(0,inplace=True) #fill NaNs with 0 otherwise it will be ignored by render_shapes method
    return df

    

def function_identify_cluster_corr(sdata,cluster_membership_df,group,bin_size=8):
    table = sdata.tables[f"square_00{bin_size}um"]
    spots_with_expression = table.obs[table.obs[group] != 0].index

    result_df = table[spots_with_expression,].to_df().apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_corr(row, cluster_membership_df)}), axis=1)
   

    others_df= pd.DataFrame({'Index': list(set(table.obs.index) - set(spots_with_expression)), 'assigned_cluster': [None]*len(set(table.obs.index) - set(spots_with_expression))})

    df=pd.concat([result_df,others_df])
    df.set_index('Index', inplace=True)
    df[f'{group}_clusters'] = pd.Categorical(df['assigned_cluster'],categories=cluster_membership_df.index.unique())
    df.drop(columns=['assigned_cluster'],inplace=True)
    return df

def function_identify_cluster_cosine(sdata,cluster_membership_df,group,bin_size=8):

    table = sdata.tables[f"square_00{bin_size}um"]
    spots_with_expression = table.obs[table.obs[group] != 0].index

    result_df = table[spots_with_expression,].to_df().apply(lambda row: pd.Series({'Index': row.name, 'assigned_cluster': function_row_cosine(row, cluster_membership_df)}), axis=1)
   

    others_df= pd.DataFrame({'Index': list(set(table.obs.index) - set(spots_with_expression)), 'assigned_cluster': [None]*len(set(table.obs.index) - set(spots_with_expression))})

    df=pd.concat([result_df,others_df])
    df.set_index('Index', inplace=True)
    df[f'{group}_clusters'] = pd.Categorical(df['assigned_cluster'],categories=cluster_membership_df.index.unique())
    df.drop(columns=['assigned_cluster'],inplace=True)
    return df

def function_row_corr(row,markers_df):
    a={}
    for c in markers_df.index.unique():
        vector_series=pd.Series(markers_df[["names","logfoldchanges"]].loc[c]["logfoldchanges"].values, index=markers_df[["names","logfoldchanges"]].loc[c]["names"].values)
        vector_series=vector_series.reindex(row.index,fill_value=np.nan)

        a[c]=spearmanr(row, vector_series,nan_policy="omit",axis=1)[0]

    
    return str(max(a, key=a.get))

def function_row_cosine(row, markers_df):
    a = {}
    for c in markers_df.index.unique():
        vector_series = pd.Series(markers_df[["names", "logfoldchanges"]].loc[c]["logfoldchanges"].values, index=markers_df[["names", "logfoldchanges"]].loc[c]["names"].values)
        vector_series = vector_series.reindex(row.index, fill_value=np.nan)

        # Calculate cosine distance and handle cases where vectors might be all NaNs after reindexing
        if not vector_series.isnull().all() and not row.isnull().all():
            a[c] = cosine(row.fillna(0), vector_series.fillna(0))
        else:
            a[c] = np.nan  # Assign NaN if either vector is all NaNs

    # Return the key with the minimum cosine distance (maximum similarity)
    # Filter out NaN values from the dictionary
    filtered_a = {k: v for k, v in a.items() if not np.isnan(v)}
    return str(min(filtered_a, key=filtered_a.get)) if filtered_a else None



def test_function():
    print("Hello World Now")


