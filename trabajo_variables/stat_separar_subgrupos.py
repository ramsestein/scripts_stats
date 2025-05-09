import pandas as pd

# Cargar el archivo Excel original
archivo_origen = 'iProve_comp_resp.xlsx'  # Cambia el nombre si es necesario
df = pd.read_excel(archivo_origen)

df_airtestdico_0 = df[(df['spO2pre'] < 97)]
df_airtestdico_1 = df[(df['spO2pre'] >= 97)]


# Guardar los datos en archivos Excel separados
df_airtestdico_0.to_excel('PPC_enf.xlsx', index=False)
df_airtestdico_1.to_excel('PPC_sano.xlsx', index=False)

print("Archivos generados:")
