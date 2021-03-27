from category_encoders import one_hot, ordinal
import pandas as pd
import datetime
from functools import reduce

# 7. Attribute information:
#
#    For more information, read [Moro et al., 2011].
#
#    Input variables:
#    # bank client data:
#    1 - age (numeric) - quantitativa discreta
#    2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services") - qualitativa nominal
#    3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed) - qualitativa nominal
#    4 - education (categorical: "unknown","secondary","primary","tertiary") - qualitativa ordinal
#    5 - default: has credit in default? (binary: "yes","no") - qualitativa nominal
#    6 - balance: average yearly balance, in euros (numeric) - quantitativa continua
#    7 - housing: has housing loan? (binary: "yes","no") - qualitativa nominal
#    8 - loan: has personal loan? (binary: "yes","no") - qualitativa nominal
#    # related with the last contact of the current campaign:
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular") - qualitativa nominal
#   10 - day: last contact day of the month (numeric) - quantitativa discreta
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec") - quantitativa discreta
#   12 - duration: last contact duration, in seconds (numeric) - quantitativa continua
#    # other attributes:
#   13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact) - quantitativa discreta
#   14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted) - quantitativa discreta
#   15 - previous: number of contacts performed before this campaign and for this client (numeric) - quantitativa discreta
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success") - qualitativa nominal
#
#   Output variable (desired target):
#   17 - y - has the client subscribed a term deposit? (binary: "yes","no") - qualitativa nominal
#
# 8. Missing Attribute Values: None


print('Iniciando...')
pd.set_option("display.max_columns", 150)
dir_csv = '...\\bank.csv'
dict_col_education = {'unknown': 0, 'primary': 1, 'secondary': 2, 'tertiary': 3}

df_treino = pd.read_csv(dir_csv, ';', skip_blank_lines=True, encoding='utf-8')
col_target = ['y']

# -----------
# 0 - define os tipos de variaveis
# -----------
cols_numericas = ['age', 'day', 'campaign', 'pdays', 'previous', 'balance', 'duration']
cols_nao_numericas = ['y', 'education', 'month']
cols_pivoteadas = ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'poutcome']

# ---------
# 2 - aplica os Encoders
# ---------

# one hot
df_onehotencoded = one_hot.OneHotEncoder(cols=cols_pivoteadas, use_cat_names=True)\
    .fit_transform(df_treino[cols_pivoteadas])
df_onehotencoded['source_ix'] = df_onehotencoded.index
# deleta coluna com ponto no nome
#df_onehotencoded['job_admin'] = df_onehotencoded['job_admin.']
#del df_onehotencoded['job_admin.']

# ordinal
df_normalizar_nao_numericas = df_treino[cols_nao_numericas].copy()

for col in cols_nao_numericas:
    try:
        print(f'Ordenando coluna "{col}"')
        df_tmp = df_normalizar_nao_numericas.copy()
        col_ord = col + '_ordenada'

        # indica que é o primeiro loop - guarda o indice original para posterior conferencia
        if col == cols_nao_numericas[0]:
            df_tmp['source_ix'] = df_tmp.index

        # ordena pela coluna e pega os valores unicos
        distintos = pd.DataFrame(df_tmp[col].sort_values(ascending=True)
                                 .drop_duplicates()).reset_index(drop=True)

        # da um n. sequencial pra coluna
        if col == 'month':
            distintos[col_ord] = distintos[col].apply(lambda l: datetime.datetime.strptime(l, '%b').month)
        elif col == 'education':
            distintos[col_ord] = distintos[col].apply(lambda l: dict_col_education[l])
        else:
            distintos[col_ord] = distintos.index

        # junta os 2 df -> 1 pra muitos ('1:m')
        df_merged = pd.merge(distintos, df_tmp, left_on=col, right_on=col, validate='1:m')
        # substitui a coluna pela col ordenada/numerica
        df_merged[col] = df_merged[col_ord]
        # deleta a col ordenada/numerica
        del df_merged[col_ord]
        # retroalimenta nosso df original
        df_normalizar_nao_numericas = df_merged.copy()

    except Exception as ex:
        print(str(ex))
        raise

# ---------
# 3 - aplica a normalização das variaveis - para nosso caso [sigmoide], entre 0 e 1
# ---------
# nao numericas
df_normalizado_nao_numericas = (df_normalizar_nao_numericas[cols_nao_numericas] - df_normalizar_nao_numericas[cols_nao_numericas].min()) \
                 / (df_normalizar_nao_numericas[cols_nao_numericas].max() - df_normalizar_nao_numericas[cols_nao_numericas].min())
df_normalizado_nao_numericas['source_ix'] = df_normalizar_nao_numericas['source_ix']

# numericas
# df_normalizado_numericas = (df_treino[cols_numericas] - df_treino[cols_numericas].min()) \
#                  / (df_treino[cols_numericas].max() - df_treino[cols_numericas].min())
# df_normalizado_numericas['source_ix'] = df_treino.index

df_age = (df_treino['age'] - 18) / (95 - 18)
df_cam = (df_treino['campaign'] - 1) / (63 - 1)
df_pre = (df_treino['previous'] - 0) / (275 - 0)
df_bal = (df_treino['balance'] - -8019) / (102127 - -8019)
df_dur = (df_treino['duration'] - 0) / (4918 - 0)
df_pda = (df_treino['pdays'] - -1) / (871 - -1)
df_day = (df_treino['day'] - 1) / (31 - 1)

df_numericas_normalizadas = pd.concat([df_age, df_cam, df_pre, df_bal, df_dur, df_pda, df_day], axis=1)
df_numericas_normalizadas['source_ix'] = df_treino.index

# ---------
# 4 - junta os dois dataframes trabalhados
# ---------
df_final = pd.merge(df_numericas_normalizadas, df_onehotencoded, on='source_ix', validate='1:1', how='inner')
df_final = pd.merge(df_final, df_normalizado_nao_numericas, on='source_ix', validate='1:1', how='inner')

# arredonda variaveis double para 4 decimais
double_cols = df_final.select_dtypes(include='float64')
df_final[double_cols.columns] = double_cols.round(decimals=4)

# renomeia coluna source_ix
df_final.rename(columns={'source_ix': '__REMOVER_SOURCE_IX__'}, inplace=True)

# reorganiza as colunas do dataframe pra melhor visualizacao
df_final = df_final.reindex(sorted(df_final.columns), axis=1)
sorted_nao_numericas = sorted(cols_nao_numericas)
sorted_numericas = sorted(cols_numericas)

cols1 = ['__REMOVER_SOURCE_IX__'] + sorted_numericas + [col for col in sorted_nao_numericas if col != col_target[0]]
colunas_reordenadas = (['__REMOVER_SOURCE_IX__'] + sorted_numericas +
                       [col for col in sorted_nao_numericas if col != col_target[0]] +
                       [col for col in df_final if col not in cols1])

df_final = df_final[colunas_reordenadas]
df_final[col_target] = df_final[col_target].astype('int16')
del df_final['__REMOVER_SOURCE_IX__']

print('Qtd. df_treino: ', len(df_treino), '. Qtd. df_treino_final: ', len(df_final))
if len(df_treino) != len(df_final):
    raise NameError('o dataframe de entrada e de saida devem ter o mesmo n. de registros')

# ---------
# 5 - grava o arquivo de saida
# ---------
df_final.to_csv(dir_csv.replace('.csv', '_normalizado_v03.csv'), sep=';', index=False, mode='w', encoding='utf-8')

print('Finalizando o processo...')
