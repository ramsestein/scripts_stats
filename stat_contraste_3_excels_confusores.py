import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import fisher_exact, chi2_contingency
from statsmodels.formula.api import logit

# Cargar los dos excels
high_df = pd.read_excel("PPC_all_0.xlsx")
low_df = pd.read_excel("PPC_all_1.xlsx")

# Reemplazar espacios en nombres de columnas para compatibilidad
high_df.columns = high_df.columns.str.replace(" ", "_")
low_df.columns = low_df.columns.str.replace(" ", "_")
high_df.columns = high_df.columns.str.replace(r'[^\w]', '_', regex=True)
low_df.columns = low_df.columns.str.replace(r'[^\w]', '_', regex=True)

# Combinar los DataFrames y añadir columna de grupo
combined_df = pd.concat([high_df, low_df], ignore_index=True)
combined_df['group'] = [0]*len(high_df) + [1]*len(low_df)

# Variables y confusores
columns_of_interest = [
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
    'primeraRMA_Crs', 'primeraRMA_OL_PEEP', 'SiguientesRMA_40min', 'siguientesRMA_40min', 
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

confounders = ["edad", "genero", "altura", "IMC", "ASA", "spO2pre", "diabetesMellitusII", "DM_I_o_II", "fumador", "exFumador"]

# Función para realizar ANCOVA
def perform_ancova(df, dependent_var, group_var, confounders):
    formula = f"{dependent_var} ~ C({group_var}) + {' + '.join(confounders)}"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table

# Función para realizar Bootstrap
def perform_bootstrap(df, dependent_var, group_var, confounders, n_bootstrap=1000):
    bootstrap_results = []
    for _ in range(n_bootstrap):
        boot_df = df.sample(frac=1, replace=True)
        model, _ = perform_ancova(boot_df, dependent_var, group_var, confounders)
        bootstrap_results.append(model.params.get(f"C({group_var})[T.1]", np.nan))
    bootstrap_results = [res for res in bootstrap_results if not np.isnan(res)]
    return np.mean(bootstrap_results), np.percentile(bootstrap_results, [2.5, 97.5])

# Realizar análisis
results_numeric = []
results_categoric = []

for var in columns_of_interest:
    if var not in combined_df.columns:
        print(f"⚠️ {var} no encontrada, se omite.")
        continue

    datos1 = combined_df[combined_df['group'] == 0][var].dropna()
    datos2 = combined_df[combined_df['group'] == 1][var].dropna()

    # Mínimo de datos para análisis estadístico
    if len(datos1) < 3 or len(datos2) < 3:
        print(f"Datos insuficientes en {var}.")
        continue

    print(f"Analizando: {var}")

    try:
        if combined_df[var].dtype in ['object', 'category'] or combined_df[var].dropna().isin([0, 1]).all():
            tabla = pd.crosstab(combined_df[var], combined_df['group'])

            if tabla.shape != (2, 2):
                print(f"Chi2/Fisher no posible (no 2x2) en {var}")
                continue

            if (tabla.values < 5).any():
                # Fisher exacto
                oddsratio, p_valor = fisher_exact(tabla)
                prueba = "Fisher Exact"
                estadistico = oddsratio
            else:
                # Chi-cuadrado
                chi2, p_valor, _, _ = chi2_contingency(tabla)
                prueba = "Chi-Cuadrado"
                estadistico = chi2

            results_categoric.append({
                "Variable": var,
                "Estadístico": estadistico,
                "p-valor": p_valor,
                "Prueba": prueba
            })

        else:
            # ANCOVA estrictamente para variables numéricas
            formula = f'Q("{var}") ~ C(group) + ' + " + ".join([f'Q("{c}")' for c in confounders])
            model = ols(formula, data=combined_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            f_valor = anova_table.loc["C(group)", "F"]
            p_valor = anova_table.loc["C(group)", "PR(>F)"]

            # Bootstrap sencillo
            boot_means = []
            for i in range(1000):
                sample = combined_df.sample(frac=1, replace=True)
                m1 = sample[sample["group"] == 0][var].mean()
                m2 = sample[sample["group"] == 1][var].mean()
                boot_means.append(m1 - m2)

            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

            results_numeric.append({
                "Variable": var,
                "ANCOVA F": f_valor,
                "ANCOVA p-valor": p_valor,
                "Bootstrap Mean Diff": np.mean(boot_means),
                "Bootstrap CI Lower": ci_lower,
                "Bootstrap CI Upper": ci_upper
            })

    except Exception as e:
        print(f"Error en {var}: {e}")

# Guardar resultados finales robustamente
with pd.ExcelWriter("Resultados_completos.xlsx") as writer:
    if results_numeric:
        pd.DataFrame(results_numeric).to_excel(writer, "ANCOVA_Bootstrap", index=False)
    if results_categoric:
        pd.DataFrame(results_categoric).to_excel(writer, sheet_name="Chi2_Fisher", index=False)

print("✅ Resultados generados con éxito.")
