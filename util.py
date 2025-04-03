

import numpy as np


def filter_dataset(dfx, dfy, metadata):
    '''
    Função para permitir carregar os dados no treinamento e no teste da mesma forma.
    Usa os metadados para filtrar o dataset, remover colunas e selecionar o atributo de saída (alvo).
    '''
    subdataset = metadata['subdataset']
    target_col = metadata['target']
    columns_to_drop = metadata['dropped_features']

    assert subdataset in ['geral', 'lula', 'bolsonaro']

    if subdataset == 'geral':
        # pode remover uma das colunas de candidato, porque é redundante 
        # PORÉM, não estou removendo!
        #dfx.drop(columns=['Candidato_Bolsonaro'], inplace=True)
        print('Dataset GERAL!')

    else:
        lula_indicator = int(subdataset == 'lula')  # 1 para Lula, 0 para Bolsonaro
        candidate_filter = (dfx['Candidato_Lula'] == lula_indicator)

        dfx = dfx[candidate_filter]
        dfx = dfx.drop(columns=['Candidato_Lula', 'Candidato_Bolsonaro'])

        dfy = dfy[candidate_filter]
        print('Dataset', subdataset.upper(), f'(indicador {lula_indicator})')

    dfx = dfx.drop(columns=columns_to_drop)
    
    return dfx, dfy[target_col]


# função para salvar os resultados e os parâmetros dos modelos
def save_results(filename, resultados, metadata):
    np.save(filename, [metadata, resultados], allow_pickle=True)


# função para carregar os resultados e os parâmetros dos modelos
def load_results(filename):
    metadata, r = np.load(filename, allow_pickle=True)
    #print("Carregado arquivo de resultados com descrição:")
    #print(metadata)
    return metadata, r
