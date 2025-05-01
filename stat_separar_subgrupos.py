import pandas as pd

# Cargar el archivo Excel original
archivo_origen = 'iProve_comp_resp.xlsx'  # Cambia el nombre si es necesario
df = pd.read_excel(archivo_origen)

# Filtrar los datos según las condiciones
#df_airtestdico_0 = df[(df['airTestpo1tivo180min'] == 0) & (df['spO2pre'] < 97)]
#df_airtestdico_0 = df_airtestdico_0[(df_airtestdico_0['airTestpo1tivo15min'] == 0) | (df_airtestdico_0['airTest_SpO2'] >= 97) | (df_airtestdico_0['SpO2_0'] >= 97)]###
#
#df_airtestdico_1 = df[(df['airTestpo1tivo180min'] == 1) & (df['spO2pre'] < 97)]
#df_airtestdico_1 = df_airtestdico_1[(df_airtestdico_1['airTestpo1tivo15min'] =#= 1) | (df_airtestdico_1['airTest_SpO2'] < 97) | (df_airtestdico_1['SpO2_0'] < 97)]#
#
#df_airtestdico_2 = df[(df['airTestpo1tivo180min'] == 0) & (df['spO2pre'] < 97)]
#df_airtestdico_2 = df_airtestdico_2[(df_airtestdico_2['airTestpo1tivo15min'] == 1) | (df_airtestdico_2['airTest_SpO2'] < 97) | (df_airtestdico_2['SpO2_0'] < 97)]#
#
#df_airtestdico_3 = df[(df['airTestpo1tivo180min'] == 1) & (df['spO2pre'] < 97)]
#df_airtestdico_3 = df_airtestdico_3[(df_airtestdico_3['airTestpo1tivo15min'] == 0) | (df_airtestdico_3['airTest_SpO2'] >= 97) | (df_airtestdico_3['SpO2_0'] >= 97)]
df_airtestdico_0 = df[(df['spO2pre'] < 97)]
df_airtestdico_1 = df[(df['spO2pre'] >= 97)]


# Guardar los datos en archivos Excel separados
df_airtestdico_0.to_excel('PPC_enf.xlsx', index=False)
df_airtestdico_1.to_excel('PPC_sano.xlsx', index=False)
#df_airtestdico_2.to_excel('PPC_enf_airTestmal_a_bien.xlsx', index=False)
#df_airtestdico_3.to_excel('PPC_enf_airTestbien_a_mal.xlsx', index=False)
#df_airtestdico_1_rest.to_excel('AirtestDico_1_rescate_1.xlsx', index=False)

print("Archivos generados:")
print("1. AirtestDico_0.xlsx")
print("2. AirtestDico_1_rescate_0.xlsx")
print("3. AirtestDico_1_rescate_1.xlsx")
