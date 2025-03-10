

import numpy as np


def filter_dataset(dfx, dfy, subdataset, target_col):
    assert subdataset in ['geral', 'lula', 'bolsonaro']

    if subdataset == 'geral':
        # pode remover uma das colunas de candidato, porque é redundante 
        #dfx.drop(columns=['Candidato_Bolsonaro'], inplace=True)
        # PORÉM, não estou removendo!
        print('Dataset GERAL!')

    else:
        lula_indicator = int(subdataset == 'lula')  # 1 para Lula, 0 para Bolsonaro
        candidate_filter = (dfx['Candidato_Lula'] == lula_indicator)

        dfx = dfx[candidate_filter]
        dfx = dfx.drop(columns=['Candidato_Lula', 'Candidato_Bolsonaro'])

        dfy = dfy[candidate_filter]
        print('Dataset', subdataset.upper(), f'(indicador {lula_indicator})')
    
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
