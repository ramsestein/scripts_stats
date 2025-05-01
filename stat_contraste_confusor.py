import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.utils import resample
import os

# Cargar los datos desde el archivo Excel
input_file = "iProve_comp_resp.xlsx"
df = pd.read_excel(input_file)

columnas_existentes = [
    'días_de_estancia', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'pAfI', 'falloHemodinamico', 
    'hipertensionEndocraneal', 'neumotorax', 'tipoProcedimiento', 'tipoCirugia', 'esCirugiaOncologica', 
    'diagnosticoPrimario', 'aRISCAT', 'ClinicalFrailityScale', 'hbPreoperatoria', 'indiceCHARLSON', 'apfel', 
    'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'cardiopatiaIsquemica', 'consumoAlcohol', 
    'dislipemia', 'ePOC', 'insuficienciaRenal', 'insuficienciaHepatica', 'enfermedadNeuromuscular', 
    'apneaObstructivaSueno', 'oncologico', 'inmunosupresion', 'Otras_enfermedades', 'roncaFuertemente', 
    'decaimiento', 'dejaRespirarSueno', 'tratamientoTensionAlta', 'imcAlto', 'esMayor50', 
    'circunferenciaCuelloAlta', 'generoPaciente', 'usoAntibioticos', 'antiHTA', 'aspirina', 'estatinas', 
    'antidiabeticosOrales', 'insulina', 'inhaladoresConCorticoides', 'inhaladores1nCorticoides', 
    'corticoides', 'quimioterapiaPreviaCirugia', 'radioterapiaPreviaCirugia', 'pH', 'SatO2_preOp', 
    'peepBasal', 'peepT1', 'peepT2', 'peepT3', 'peepMedia', 'd_peep', 'peepFinal', 'frBasal', 'frT1', 
    'frT2', 'frT3', 'frMedia', 'frFinal', 'vtBasal', 'vtT1', 'vtT2', 'vtT3', 'vtMedia', 'vtFinal', 
    'spo2Basal', 'spo2T1', 'spo2T2', 'spo2T3', 'spo2Media', 'spo2Final', 'fio2Basal', 'fio2T1', 'fio2T2', 
    'd_PaFi', 'fio2T3', 'd_SaFi', 'fio2Media', 'PaFi_Final', 'PaFI_Media', 'fio2Final', 'SaFi_Final', 
    'SaFi_Media', 'SaFi_Basal', 'PaFi_Basal', 'pao2Basal', 'pao2T1', 'pao2T2', 'pao2T3', 'pao2Media', 
    'pao2Final', 'pafiBasal', 'pafiMedia', 'pafiFinal', 'paco2Basal', 'paco2T1', 'paco2T2', 'paco2T3', 
    'paco2Media', 'paco2Final', 'pHBasal', 'pHT1', 'pHT2', 'pHT3', 'pHMedia', 'pHFinal', 'oriBasal', 
    'oriMedia', 'oriFinal', 'indicePerfusionBasal', 'indicePerfusionMedia', 'indicePerfusionFinal', 
    'pviBasal', 'pviMedia', 'pviFinal', 'sphbBasal', 'sphbMedia', 'sphbFinal', 'presionPicoBasal', 
    'presionPicoT1', 'presionPicoT2', 'presionPicoT3', 'presionPicoMedia', 'presionPicoFinal', 
    'presionMesetaBasal', 'presionMesetaT1', 'presionMesetaT2', 'presionMesetaT3', 'presionMesetaMedia', 
    'presionMesetaFinal', 'driving_preBasal', 'crsBasal', 'crsT1', 'crsT2', 'crsT3', 'crsMedia', 'crsFinal', 
    'mejorcrs', 'pamBasal', 'pamT1', 'pamT2', 'pamT3', 'pamMedia', 'pamFinal', 'icBasal', 'icT1', 'icT2', 
    'icT3', 'icMedia', 'icFinal', 'vVSvPPBasal', 'vVSvPPMedia', 'vVSvPPFinal', 'tEsogagicaBasal', 
    'tEsogagicaMedia', 'tEsogagicaFinal', 'glucemiaBasal', 'glucemiaMedia', 'glucemiaFinal', 'volumen_total', 
    'cristaloides', 'cHematies', 'coloides', 'diuresis', 'perdidaSangreEstimada', 'duracionCirugia', 
    'duracionVM', 'duracionOLV', 'duracionOLV_TLV', 'conver1onLaparotomia', 'usoFarmacosVasoactivos', 
    '0radrenalina', 'dobutamina', 'efedrina', 'fenilefrina', 'vasopre1na', 'hipnotico', 'rnm', 
    'rever1onRNM', 'reversionRNMTexto', 'monitorizacionTOF', 'TOFr>0,9BE', 'analgesia', 'epidural2', 'sng', 
    'sngTexto', 'retiradasngpreextot', 'profilaxisPONV', 'profilaxisPONVTexto', 'profilaxisPONVTexto2', 
    'primeraRMA_Crs', 'primeraRMA_OL_PEEP', 
    'mraDesconexionAccidental', 'mraDesconexionAccidentalTexto', 'mraDesconexionAccidentalnumeroveces', 
    'presionaperturamax', 'volumencorrienteunipulmonar', 'driving_pre_postMRA', 'fracasoMRA_primeraMRA', 
    'fracasoMRA_primeraMRA_causa', 'fracasoMRA_primeraMRA_Efedrina_Fenilafrina', 'fracasoMRA_1guientesMRA', 
    'fracasoMRA_1guientesMRA_Efedrina_Fenilafrina', 'numeroMRAnecesarias', 'numeroMRAfinalizadas', 
    'numeroMRAfracasadas', 'maniobraRescateIntraoperatorio', 'maniobraRescateIntraoperatorio_individperi+intra', 
    'maniobraRescateIntraoperatorio_estandar+estandarCPAPpost', 'hipoxemia', 'gafasNasales', 
    'maniobraRescatePostoperatorio', 'maniobraRescatePostoperatorioTexto', 'RescPostop_individperi15min', 
    'RescPostop_individperi', 'RescPostop_individintraCPAPpost', 'RescPostop_estandar', 
    'RescPostop_estandarCPAPpost', 'pacienteExtubadoQuirofa0', 'causaPacienteExtubadoQuirofano', 
    'causaPacienteExtubadoQuirofanoTexto', 'ingresoUci0PrevistoVM', 'tiempoVMHastaExtubacion_(minutos)', 
    'manejoPostqxSegunProtocolo', 'farmacoAnalge1a', 'airTest_SpO2', 'tiempoDesdeEntradaURPA', 
    'airTestpo1tivo15min', 'Hipoxemia15min', 'inestabilidadHemodinamica', 'inestabilidadHemodinamicaConArritia', 
    'neumotorax4', 'incumplimientoProtocoloVentilatorio', 'gasometriaPostoperatoria_SpO2', 
    'gasometriaPostoperatoria_PaO2', 'gasometria_Post_FiO2', 'gasometriaPostoperatoria_PaO2_FIO2', 
    'gasometriaPostoperatoria_PaCO2', 'gasometriaPostoperatoria_pH'
]

# Filtra únicamente las columnas que sí existen en tu DataFrame
columnas_existentes = [col for col in columnas_existentes if col in df.columns]

# Eliminar filas con valores faltantes en las variables clave
df = df.dropna(subset=columnas_existentes,thresh=5)

# Reemplazar espacios en nombres de columnas para compatibilidad con patsy
df = df.rename(columns=lambda x: x.replace(" ", "_"))

# Variables independientes, dependientes y confusores
dependent_vars = [
    'días_de_estancia', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'pAfI', 'falloHemodinamico', 
    'hipertensionEndocraneal', 'neumotorax', 'tipoProcedimiento', 'tipoCirugia', 'esCirugiaOncologica', 
    'diagnosticoPrimario', 'aRISCAT', 'ClinicalFrailityScale', 'hbPreoperatoria', 'indiceCHARLSON', 'apfel', 
    'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'cardiopatiaIsquemica', 'consumoAlcohol', 
    'dislipemia', 'ePOC', 'insuficienciaRenal', 'insuficienciaHepatica', 'enfermedadNeuromuscular', 'oncologico', 'inmunosupresion', 'Otras_enfermedades', 'roncaFuertemente', 
    'decaimiento', 'dejaRespirarSueno', 'tratamientoTensionAlta', 'imcAlto', 'esMayor50', 
    'circunferenciaCuelloAlta', 'usoAntibioticos', 'antiHTA', 'aspirina', 'estatinas', 
    'antidiabeticosOrales', 'insulina', 'inhaladoresConCorticoides', 'inhaladores1nCorticoides', 
    'corticoides', 'quimioterapiaPreviaCirugia', 'radioterapiaPreviaCirugia', 'pH', 'SatO2_preOp', 
    'peepBasal', 'peepT1', 'peepT2', 'peepT3', 'peepMedia', 'd_peep', 'peepFinal', 'frBasal', 'frT1', 
    'frT2', 'frT3', 'frMedia', 'frFinal', 'vtBasal', 'vtT1', 'vtT2', 'vtT3', 'vtMedia', 'vtFinal', 
    'spo2Basal', 'spo2T1', 'spo2T2', 'spo2T3', 'spo2Media', 'spo2Final', 'fio2Basal', 'fio2T1', 'fio2T2', 
    'd_PaFi', 'fio2T3', 'd_SaFi', 'fio2Media', 'PaFi_Final', 'PaFI_Media', 'fio2Final', 'SaFi_Final', 
    'SaFi_Media', 'SaFi_Basal', 'PaFi_Basal', 'pao2Basal', 'pao2T1', 'pao2T2', 'pao2T3', 'pao2Media', 
    'pao2Final', 'pafiBasal', 'pafiMedia', 'pafiFinal', 'paco2Basal', 'paco2T1', 'paco2T2', 'paco2T3', 
    'paco2Media', 'paco2Final', 'pHBasal', 'pHT1', 'pHT2', 'pHT3', 'pHMedia', 'pHFinal', 'oriBasal', 
    'oriMedia', 'oriFinal', 'indicePerfusionBasal', 'indicePerfusionMedia', 'indicePerfusionFinal', 
    'pviBasal', 'pviMedia', 'pviFinal', 'sphbBasal', 'sphbMedia', 'sphbFinal', 'presionPicoBasal', 
    'presionPicoT1', 'presionPicoT2', 'presionPicoT3', 'presionPicoMedia', 'presionPicoFinal', 
    'presionMesetaBasal', 'presionMesetaT1', 'presionMesetaT2', 'presionMesetaT3', 'presionMesetaMedia', 
    'presionMesetaFinal', 'driving_preBasal', 'crsBasal', 'crsT1', 'crsT2', 'crsT3', 'crsMedia', 'crsFinal', 
    'mejorcrs', 'pamBasal', 'pamT1', 'pamT2', 'pamT3', 'pamMedia', 'pamFinal', 'icBasal', 'icT1', 'icT2', 
    'icT3', 'icMedia', 'icFinal', 'vVSvPPBasal', 'vVSvPPMedia', 'vVSvPPFinal', 'tEsogagicaBasal', 
    'tEsogagicaMedia', 'tEsogagicaFinal', 'glucemiaBasal', 'glucemiaMedia', 'glucemiaFinal', 'volumen_total', 
    'cristaloides', 'cHematies', 'coloides', 'diuresis', 'perdidaSangreEstimada', 'duracionCirugia', 
    'duracionVM', 'duracionOLV', 'duracionOLV_TLV', 'conver1onLaparotomia', 'usoFarmacosVasoactivos', 
    '0radrenalina', 'dobutamina', 'efedrina', 'fenilefrina', 'vasopre1na', 'hipnotico', 'rnm', 
    'rever1onRNM', 'reversionRNMTexto', 'monitorizacionTOF', 'analgesia', 'epidural2', 'sng', 
    'sngTexto', 'retiradasngpreextot', 'profilaxisPONV', 'profilaxisPONVTexto', 'profilaxisPONVTexto2', 
    'primeraRMA_Crs', 'primeraRMA_OL_PEEP', 
    'mraDesconexionAccidental', 'mraDesconexionAccidentalTexto', 'mraDesconexionAccidentalnumeroveces', 
    'presionaperturamax', 'volumencorrienteunipulmonar', 'driving_pre_postMRA', 'fracasoMRA_primeraMRA', 
    'fracasoMRA_primeraMRA_causa', 'fracasoMRA_primeraMRA_Efedrina_Fenilafrina', 'fracasoMRA_1guientesMRA', 
    'fracasoMRA_1guientesMRA_Efedrina_Fenilafrina', 'numeroMRAnecesarias', 'numeroMRAfinalizadas', 
    'numeroMRAfracasadas', 'maniobraRescateIntraoperatorio', 'hipoxemia', 'gafasNasales', 
    'maniobraRescatePostoperatorio', 'maniobraRescatePostoperatorioTexto', 'RescPostop_individperi15min', 
    'RescPostop_individperi', 'RescPostop_individintraCPAPpost', 'RescPostop_estandar', 
    'RescPostop_estandarCPAPpost', 'pacienteExtubadoQuirofa0', 'causaPacienteExtubadoQuirofano', 
    'causaPacienteExtubadoQuirofanoTexto', 'tiempoVMHastaExtubacion_(minutos)', 
    'manejoPostqxSegunProtocolo', 'farmacoAnalge1a', 'airTest_SpO2', 'tiempoDesdeEntradaURPA', 
    'airTestpo1tivo15min', 'Hipoxemia15min', 'inestabilidadHemodinamica', 'inestabilidadHemodinamicaConArritia', 
    'neumotorax4', 'incumplimientoProtocoloVentilatorio', 'gasometriaPostoperatoria_SpO2', 
    'gasometriaPostoperatoria_PaO2', 'gasometria_Post_FiO2', 'gasometriaPostoperatoria_PaO2_FIO2', 
    'gasometriaPostoperatoria_PaCO2', 'gasometriaPostoperatoria_pH', "edad", "genero", "altura", "IMC", "ASA", "spO2pre", 
    "diabetesMellitusII", "DM_I_o_II", "fumador", "exFumador", "PPC_all"
]
confounders = ["edad", "genero", "altura", "IMC", "ASA", "spO2pre", "diabetesMellitusII", "DM_I_o_II", "fumador", "exFumador"]
group_var = "PPC_all"

# Crear directorio para guardar resultados
output_file = "results_PPC_conf.xlsx"
if os.path.exists(output_file):
    os.remove(output_file)

# Función para realizar ANCOVA
def perform_ancova(df, dependent_var, group_var, confounders):
    formula = f"{dependent_var} ~ C({group_var}) + {' + '.join(confounders)}"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    return model, anova_table

# Función para realizar bootstrap
def perform_bootstrap(df, dependent_var, group_var, confounders, n_iterations=1000):
    boot_results = []
    for i in range(n_iterations):
        # Resample data
        boot_df = resample(df, replace=True, random_state=i)
        model, _ = perform_ancova(boot_df, dependent_var, group_var, confounders)
        # Guardar el coeficiente y p-valor del grupo
        coef = model.params.get(f"C({group_var})[T.1]", np.nan)
        p_value = model.pvalues.get(f"C({group_var})[T.1]", np.nan)
        boot_results.append({"Iteration": i + 1, "Coef": coef, "P-Value": p_value})
    return pd.DataFrame(boot_results)


# Inicializar listas para almacenar resultados
results_ancova = []
results_chi2 = []
results_bootstrap = []

# Aquí haces los análisis para cada variable de interés
for var in dependent_vars:
    print(f"Analizando: {var}")

    # Verificar si la variable tiene datos suficientes
    if df[var].dropna().empty or df[group_var].dropna().empty:
        print(f"No hay datos suficientes para analizar {var}.")
        continue

    # Verificar variabilidad
    if df[var].nunique() <= 1:
        print(f"No hay variabilidad suficiente en {var}.")
        continue

    try:
        # ANCOVA
        model, anova_table = perform_ancova(df, var, group_var, confounders)
        anova_table['Variable'] = var
        results_ancova.append(anova_table.reset_index())

        # Chi-cuadrado solo si es categórica/binaria
        if df[var].nunique() <= 10:
            tabla_contingencia = pd.crosstab(df[var], df[group_var])
            chi2, p_chi2, dof, expected = sm.stats.Table(tabla_contingencia).test_nominal_association()
            results_chi2.append({"Variable": var, "Chi2": chi2, "p-value": p_chi2})

        # Bootstrap
        boot_results_df = perform_bootstrap(df, var, group_var, confounders)
        boot_results_df['Variable'] = var
        results_bootstrap.append(boot_results_df)

    except Exception as e:
        print(f"Error analizando {var}: {e}")

# Concatenar resultados
if results_ancova:
    results_ancova = pd.concat(results_ancova, ignore_index=True)
if results_bootstrap:
    results_bootstrap = pd.concat(results_bootstrap, ignore_index=True)


with pd.ExcelWriter("resultados.xlsx") as writer:
    hojas_guardadas = False

    if isinstance(results_ancova, pd.DataFrame) and not results_ancova.empty:
        results_ancova.to_excel(writer, sheet_name="ANCOVA", index=False)
        print("✅ ANCOVA guardado.")
        hojas_guardadas = True

    if results_chi2:
        pd.DataFrame(results_chi2).to_excel(writer, sheet_name="Chi2", index=False)
        print("✅ Chi-cuadrado guardado.")
        hojas_guardadas = True

    if isinstance(results_bootstrap, pd.DataFrame) and not results_bootstrap.empty:
        results_bootstrap.to_excel(writer, sheet_name="Bootstrap", index=False)
        print("✅ Bootstrap guardado.")
        hojas_guardadas = True

    if not hojas_guardadas:
        pd.DataFrame({"Mensaje": ["No hay resultados válidos generados"]}).to_excel(writer, sheet_name="SinResultados", index=False)
        print("⚠️ No había resultados válidos, generada hoja informativa.")

print("✅ Resultados generados y guardados correctamente.")








