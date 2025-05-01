import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, shapiro, f_oneway, kruskal

# Función para comprobar normalidad con Shapiro-Wilk
def check_normality(data, columns):
    normality_results = []
    for col in columns:
        stat, p_value = shapiro(data[col].dropna())
        normality_results.append({
            'Variable': col,
            'Statistic': stat,
            'p-value': p_value,
            'Normal': p_value > 0.05
        })
    return pd.DataFrame(normality_results)

# Función para realizar contrastes de hipótesis
def perform_hypothesis_test(df, var, group_var, is_normal):
    group_data = [df[df[group_var] == g][var].dropna() for g in df[group_var].unique()]
    if len(group_data) == 2:
        if is_normal:
            stat, p_value = ttest_ind(group_data[0], group_data[1], equal_var=False)
            test = "t-test"
        else:
            stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
            test = "Mann-Whitney U"
    elif len(group_data) > 2:
        if is_normal:
            stat, p_value = f_oneway(*group_data)
            test = "ANOVA"
        else:
            stat, p_value = kruskal(*group_data)
            test = "Kruskal-Wallis"
    else:
        stat, p_value, test = None, None, "No suficientes datos"
    return {"Variable": var, "Test usado": test, "Estadístico": stat, "p-valor": p_value}

# Crear carpeta para guardar gráficas
output_dir = './img'
os.makedirs(output_dir, exist_ok=True)

# Ruta para guardar resultados
output_excel = 'PPC_normal_normalidad.xlsx'

# Cargar datos
file_path = 'iProve_comp_resp.xlsx'  # Cambiar por la ruta correcta
df = pd.ExcelFile(file_path).parse('Hoja1')

# Limpieza de datos
columns_of_interest = ['edad', 'genero','días de estancia','altura', 'peso', 'IMC', 'cirugiaAbdominalProgramada', 'cirugiaAbdominalUrgencia', 'pAfI', 'falloHemodinamico', 'hipertensionEndocraneal', 'neumotorax', 'tipoProcedimiento', 'tipoCirugia', 'esCirugiaOncologica', 'diagnosticoPrimario', 'ASA', 'aRISCAT', 'spO2pre(21)', 'ClinicalFrailityScale', 'hbPreoperatoria', 'indiceCHARLSON', 'apfel', 'infeccionRespiratoriaUltimoMes', 'hipertensionArterial', 'cardiopatiaIsquemica', 'diabetesMellitusI', 'diabetesMellitusII', 'DM I o II', 'fumador', 'exFumador', 'consumoAlcohol', 'dislipemia', 'ePOC', 'insuficienciaRenal', 'insuficienciaHepatica', 'enfermedadNeuromuscular', 'apneaObstructivaSueño', 'oncologico', 'inmunosupresion', 'Otras enfermedades', 'roncaFuertemente', 'decaimiento', 'dejaRespirarSueno', 'tratamientoTensionAlta', 'imcAlto', 'esMayor50', 'circunferenciaCuelloAlta', 'generoPaciente', 'usoAntibioticos', 'antiHTA', 'aspirina', 'estatinas', 'antidiabeticosOrales', 'insulina', 'inhaladoresConCorticoides', 'inhaladores1nCorticoides', 'corticoides', 'quimioterapiaPreviaCirugia', 'radioterapiaPreviaCirugia', 'pH', 'SatO2_preOp', 'peepBasal', 'peepT1', 'peepT2', 'peepT3', 'peepMedia', 'd_peep', 'peepFinal', 'frBasal', 'frT1', 'frT2', 'frT3', 'frMedia', 'frFinal', 'vtBasal', 'vtT1', 'vtT2', 'vtT3', 'vtMedia', 'vtFinal', 'spo2Basal', 'spo2T1', 'spo2T2', 'spo2T3', 'spo2Media', 'spo2Final', 'fio2Basal', 'fio2T1', 'fio2T2', 'd_PaFi', 'fio2T3', 'd_SaFi', 'fio2Media', 'PaFi_Final', 'PaFI_Media', 'fio2Final', 'SaFi_Final', 'SaFi_Media', 'SaFi_Basal', 'PaFi_Basal', 'pao2Basal', 'pao2T1', 'pao2T2', 'pao2T3', 'pao2Media', 'pao2Final', 'pafiBasal', 'pafiMedia', 'pafiFinal', 'paco2Basal', 'paco2T1', 'paco2T2', 'paco2T3', 'paco2Media', 'paco2Final', 'pHBasal', 'pHT1', 'pHT2', 'pHT3', 'pHMedia', 'pHFinal', 'oriBasal', 'oriMedia', 'oriFinal', 'indicePerfusionBasal', 'indicePerfusionMedia', 'indicePerfusionFinal', 'pviBasal', 'pviMedia', 'pviFinal', 'sphbBasal', 'sphbMedia', 'sphbFinal', 'presionPicoBasal', 'presionPicoT1', 'presionPicoT2', 'presionPicoT3', 'presionPicoMedia', 'presionPicoFinal', 'presionMesetaBasal', 'presionMesetaT1', 'presionMesetaT2', 'presionMesetaT3', 'presionMesetaMedia', 'presionMesetaFinal', 'driving_preBasal', 'crsBasal', 'crsT1', 'crsT2', 'crsT3', 'crsMedia', 'crsFinal', 'mejorcrs', 'pamBasal', 'pamT1', 'pamT2', 'pamT3', 'pamMedia', 'pamFinal', 'icBasal', 'icT1', 'icT2', 'icT3', 'icMedia', 'icFinal', 'vVSvPPBasal', 'vVSvPPMedia', 'vVSvPPFinal', 'tEsogagicaBasal', 'tEsogagicaMedia', 'tEsogagicaFinal', 'glucemiaBasal', 'glucemiaMedia', 'glucemiaFinal', 'volumen total', 'cristaloides', 'cHematies', 'coloides', 'diuresis', 'perdidaSangreEstimada', 's', 'duracionCirugia', 'duracionVM', 'duracionOLV', 'duracionOLV_TLV', 'conver1onLaparotomia', 'usoFarmacosVasoactivos', '0radrenalina', 'dobutamina', 'efedrina', 'fenilefrina', 'vasopre1na', 'hipnotico', 'rnm', 'rever1onRNM', 'reversionRNMTexto', 'monitorizacionTOF', 'TOFr>0,9BE', 'analgesia', 'epidural', 'sng', 'sngTexto', 'retiradasngpreextot', 'profilaxisPONV', 'profilaxisPONVTexto', 'profilaxisPONVTexto2', 'primeraRMA_Crs', 'primeraRMA_OL_PEEP', 'SiguientesRMA_40min', 'siguientesRMA_40min_Crs', 'siguientesRMA_40min_OL_PEEP', 'SiguientesRMA_80min', 'siguientesRMA_80min_Crs', 'siguientesRMA_80min_OL_PEEP', 'siguientesRMA_120min', 'siguientesRMA_120min_Crs', 'siguientesRMA_120min_OL_PEEP', 'siguientesRMA_160min', 'siguientesRMA_160min_Crs', 'siguientesRMA_160min_OL_PEEP', '1guientesRMA_200min', 'siguientesRMA_200min_Crs', 'siguientesRMA_200min_OL_PEEP', 'siguientesRMA_240min', 'siguientesRMA_240min_Crs', 'siguientesRMA_240min_OL_PEEP', 'siguientesRMA_280min', 'siguientesRMA_280min_Crs', 'siguientesRMA_280min_OL_PEEP', 'siguientesRMA_320min', 'siguientesRMA_320min_Crs', 'siguientesRMA_320min_OL_PEEP', 'mraDesconexionAccidental', 'mraDesconexionAccidentalTexto', 'mraDesconexionAccidentalnumeroveces', 'presionaperturamax', 'volumencorrienteunipulmonar', 'driving_pre_postMRA', 'fracasoMRA_primeraMRA', 'fracasoMRA_primeraMRA_causa', 'fracasoMRA_primeraMRA_Efedrina_Fenilafrina', 'fracasoMRA_1guientesMRA', 'fracasoMRA_1guientesMRA_Efedrina_Fenilafrina', 'numeroMRAnecesarias', 'numeroMRAfinalizadas', 'numeroMRAfracasadas', 'maniobraRescateIntraoperatorio', 'maniobraRescateIntraoperatorio_individperi+intra', 'maniobraRescateIntraoperatorio_estandar+estandarCPAPpost', 'hipoxemia', 'gafasNasales', 'maniobraRescatePostoperatorio', 'maniobraRescatePostoperatorioTexto', 'RescPostop_individperi15min', 'RescPostop_individperi', 'RescPostop_individintraCPAPpost', 'RescPostop_estandar', 'RescPostop_estandarCPAPpost', 'pacienteExtubadoQuirofa0', 'causaPacienteExtubadoQuirofano', 'causaPacienteExtubadoQuirofanoTexto', 'ingresoUci0PrevistoVM', 'tiempoVMHastaExtubacion (minutos)', 'manejoPostqxSegunProtocolo', 'farmacoAnalge1a', 'epidural2', 'paravertebral', 'minutosTrasCirugia_15_eva', 'minutosTrasCirugia_15_do1sFarmaco', 'minutosTrasCirugia_60_eva', 'minutosTrasCirugia_60_do1sFarmaco', 'minutosTrasCirugia_120_eva', 'minutosTrasCirugia_120_do1sFarmaco', 'minutosTrasCirugia_180_eva', 'airTest_SpO2', 'airTest_PI', 'airTest_ORI', 'airTest_PVI', 'airTest_SpHb', 'tiempoDesdeEntradaURPA', 'airTestpo1tivo15min', 'airTestpo1tivo60min', 'airTestpo1tivo120min', 'airTestpo1tivo180min', 'Hipoxemia15min', 'Hipoxemia60min', 'Hipoxemia120min', 'Hipoxemia180min', 'inestabilidadHemodinamica', 'inestabilidadHemodinamicaConArritia', 'neumotorax4', 'incumplimientoProtocoloVentilatorio', 'incumplimientoMRA', 'incumplimientoPruebaAirTest', 'incumplimientoCpap', 'incumplimientoVmni', 'incumplimientoManiobraRescateVentilatorio', 'gasometriaPostoperatoria_SpO2', 'gasometriaPostoperatoria_PaO2', 'gasometria_Post_FiO2', 'gasometriaPostoperatoria_PaO2_FIO2', 'gasometriaPostoperatoria_PaCO2', 'gasometriaPostoperatoria_pH', 'clasificacionHerida', 'indiceNNIS', 'SOFA_0', 'SpO2_0', 'FIO2_0', 'PaO2_0', 'PaFi_0', 'PaCO2_0', 'pH_0', 'FrecResp_0', 'MuscAccesoria_0', 'Atelectasia_0', 'Hipoxemia_0', 'ARDS_0', 'Neumonia_0', 'Broncoespasmo_0', 'neumotorax_0', 'derramepleural_0', 'fallorespiratorioagudo_0', 'CPAP_0', 'VMNI_0', 'VMI_0', 'TAM_0', 'FC_0', 'fallocardiaco_0', 'Isquemiacardiaca_0', 'infeccionHeridaQuirurgicaDIA0', 'dehicencia_0', 'SIRS_0', 'SEPSIS_0', 'ShockSeptico_0', 'AKI_0', 'AKI1_0', 'AKI2_0', 'AKI3_0', 'SOFA_1', 'SpO2_1', 'FIO2_1', 'PaO2_1', 'PaFi_1', 'PaCO2_1', 'pH_1', 'FrecResp_1', 'MusculaturaAccesoria_1', 'infeccionHeridaQuirurgica_1', 'infeccionHeridaQuirurgicaTexto_1', 'complicacionPulmonar_1', 'falloRespiratorioLeve_1', 'distresLeve_1', 'distresModerado_1', 'distresGrave_1', 'atelectasia_1', 'neumonitisAspirativa_1', 'falloResoiratorioGrave_1', 'InfeccionRespiratoria_1', 'neumotorax_1', 'edemaPulmonar_1', 'falloWeaning_1', 'derramePleural_1', 'broncoespasmo_1', 'embolismoPulomonar_1', 'Hipoxemia_1', 'CPAP_1', 'VMNI_1', 'VMI_1', 'PruebaImagen_1', 'RxTorax_1', 'ecografiaPulmonar_1', 'TCTorax_1', 'complicacionSistemica_1', 'ShockSeptico_1', 'Sepsis_1', 'SIRS_1', 'falloCardiaco_1', 'arritmiaDeNovo_1', 'TAM_1', 'FC_1', 'fallomultiorganico_1', 'hemorragiaPostop_1', 'Infeccionurinaria_1', 'AKI_1', 'AKI1_1', 'AKI2_1', 'AKI3_1', 'isquemiaMiocardica_1', 'delirio_1', 'ileoParalitico_1', 'FalloAnastomosis_1', 'DehiscenciaSutura_1', 'admisionUCI_1', 'UCIPorProtocolo_1', 'UCIShockSeptico_1', 'UCIsepsis_1', 'UCIFalloRenal_1', 'UCIrespiratorio_1', 'UCIFalloMultiorganico_1', 'UCIFalloHemodinamico_1', 'UCIOtros_1', 'UCIEstancia_1', 'Reintervencion_1', 'ReintervencionSangrado_1', 'ReintervencionInfeccion_1', 'ReintervencionFalloAnastomosis_1', 'ReintervencionOtros_1', 'SOFA_2', 'SpO2_2', 'FIO2_2', 'PaO2_2', 'PaFi_2', 'PaCO2_2', 'pH_2', 'FrecResp_2', 'MuscAcces_2', 'Atelectasia_2', 'Hipoxemia_2', 'ARDS_2', 'Neumonia_2', 'Broncoespasmo_2', 'Neumotorax_2', 'DerramePleural_2', 'FalloRespiratorioGrave_2', 'CPAP_2', 'VMNI_2', 'VMI_2', 'TAM_2', 'FC_2', 'FalloCardiaco_2', 'Arritmiadenovo_2', 'IsquemiaMiocardica_2', 'infeccionHeridaQuirurgica_2', 'infeccionHeridaQuirurgicaTexto_2', 'DehiscenciaSutura_2', 'SIRS_2', 'Sepsis_2', 'ShockSeptico_2', 'AKI_2', 'AKI1_2', 'AKI2_2', 'AKI3_2', 'Delirio_2', 'InfeccionHeridaQuirurgica_3', 'complicacionPulmonar_3', 'falloRespiratorioLeve_3', 'distresLeve_3', 'distresModerado_3', 'distresGrave_3', 'atelectasia_3', 'neumonitisAspirativa_3', 'falloResoiratorioGrave_3', 'InfeccionRespiratoria_3', 'Neumonia_3', 'neumotorax_3', 'edemaPulmonar_3', 'falloWeaning_3', 'derramePleural_3', 'broncoespasmo_3', 'embolismoPulomonar_3', 'PruebaImagen_3', 'RxTorax_3', 'ecografiaPulmonar_3', 'TCTorax_3', 'complicacionSistemica_3', 'ShockSeptico_3', 'Sepsis_3', 'falloCardiaco_3', 'arritmiaDeNovo_3', 'fallomultiorganico_3', 'hemorragiaPostop_3', 'Infeccionurinaria_3', 'AKI_3', 'AKI1_3', 'AKI2_3', 'AKI3_3', 'isquemiaMiocardica_3', 'delirio_3', 'ileoParalitico_3', 'FalloAnastomosis_3', 'admisionUCI_3', 'UCIPorProtocolo_3', 'UCIShockSeptico_3', 'UCIsepsis_3', 'UCIFalloRenal_3', 'UCIrespiratorio_3', 'UCIFalloMultiorganico_3', 'UCIFalloHemodinamico_3', 'UCIOtros_3', 'UCIEstancia_3', 'Reintervencion_3', 'ReintervencionSangrado_3', 'ReintervencionInfeccion_3', 'ReintervencionFalloAnastomosis_3', 'ReintervencionOtros_3', 'infeccionHeridaQuirurgica_5', 'complicacionPulmonar_5', 'falloRespiratorioLeve_5', 'distresLeve_5', 'distresModerado_5', 'distresGrave_5', 'atelectasia_5', 'neumonitisAspirativa_5', 'falloResoiratorioGrave_5', 'InfeccionRespiratoria_5', 'edemaPulmonar_5', 'falloWeaning_5', 'derramePleural_5', 'broncoespasmo_5', 'embolismoPulomonar_5', 'PruebaImagen_5', 'RxTorax_5', 'ecografiaPulmonar_5', 'TCTorax_5', 'complicacionSistemica_5', 'ShockSeptico_5', 'Sepsis_5', 'falloCardiaco_5', 'arritmiaDeNovo_5', 'fallomultiorganico_5', 'hemorragiaPostop_5', 'Infeccionurinaria_5', 'AKI_5', 'AKI1_5', 'AKI2_5', 'AKI3_5', 'isquemiaMiocardica_5', 'delirio_5', 'ileoParalitico_5', 'FalloAnastomosis_5', 'admisionUCI_5', 'UCIPorProtocolo_5', 'UCIShockSeptico_5', 'UCIsepsis_5', 'UCIFalloRenal_5', 'UCIrespiratorio_5', 'UCIFalloMultiorganico_5', 'UCIFalloHemodinamico_5', 'UCIEstancia_5', 'Reintervencion_5', 'ReintervencionSangrado_5', 'ReintervencionInfeccion_5', 'ReintervencionFalloAnastomosis_5', 'ReintervencionOtros_5', 'SOFA_7', 'SpO2_7', 'FIO2_7', 'PaO2_7', 'PaFi_7', 'PaCO2_7', 'pH_7', 'FrecResp_7', 'MuscAcces_7', 'infeccionHeridaQuirurgica_7', 'infeccionHeridaQuirurgicaTexto_7', 'complicacionPulmonar_7', 'falloRespiratorioLeve_7', 'distresLeve_7', 'distresModerado_7', 'distresGrave_7', 'atelectasia_7', 'neumonitisAspirativa_7', 'falloResoiratorioGrave_7', 'InfeccionRespiratoria_7', 'Neumonia_7', 'neumotorax_7', 'edemaPulmonar_7', 'falloWeaning_7', 'derramePleural_7', 'broncoespasmo_7', 'embolismoPulomonar_7', 'Hipoxemia_7', 'CPAP_7', 'VMNI_7', 'VMI_7', 'PruebaImagen_7', 'RxTorax_7', 'ecografiaPulmonar_7', 'TCTorax_7', 'complicacionSistemica_7', 'SIRS_7', 'ShockSeptico_7', 'Sepsis_7', 'TAM_7', 'FC_7', 'falloCardiaco_7', 'arritmiaDeNovo_7', 'IsquemiaMiocar_7', 'fallomultiorganico_7', 'hemorragiaPostop_7', 'Infeccionurinaria_7', 'AKI_7', 'AKI1_7', 'AKI2_7', 'AKI3_7', 'isquemiaMiocardica_7', 'delirio_7', 'ileoParalitico_7', 'FalloAnastomosis_7', 'DehicenciaSutura_7', 'admisionUCI_7', 'UCIPorProtocolo_7', 'UCIShockSeptico_7', 'UCIsepsis_7', 'UCIFalloRenal_7', 'UCIrespiratorio_7', 'UCIFalloMultiorganico_7', 'UCIFalloHemodinamico_7', 'UCIOtros_7', 'UCIEstancia_7', 'Reintervencion_7', 'ReintervencionSangrado_7', 'ReintervencionInfeccion_7', 'ReintervencionFalloAnastomosis_7', 'ReintervencionOtros_7', 'SOFA_30', 'SpO2_30', 'FIO2_30', 'PaO2_30', 'PaFi_30', 'PaCO2_30', 'pH_30', 'FrecResp_30', 'MuscAcces_30', 'Atelectasia_30', 'Hipoxemia_30', 'ARDS_30', 'Neumonia_30', 'InfeccionPulmonar_30', 'Broncoespasmo_30', 'Neumotorax_30', 'DerramePleural_30', 'FalloRespiratorioGrave_30', 'CPAP_30', 'VMNI_30', 'VMI_30', 'TAM_30', 'FC_30', 'FalloCardiaco_30', 'Arritmiadenovo_30', 'IsquemiaMiocardica_30', 'infeccionHeridaQuirurgica_30', 'infeccionHeridaQuirurgicaTexto_30', 'DehiscenciaSutura_30', 'SIRS_30', 'Sepsis_30', 'ShockSeptico_30', 'AKI_30', 'AKI1_30', 'AKI2_30', 'AKI3_30', 'Delirio_30', 'muscacces_general', 'atelectasia_general', 'falloresp_general', 'neumonitis_general', 'distresleve_general', 'distresmoderado_general', 'distressevero_general', 'neumonia_general', 'empiemapleural_general', 'broncoespasmo_general', 'neumotorax_general', 'fugaaerea_general', 'derramepleural_general', 'CPAP_general', 'VMNI_general', 'VMI_general', 'comppulmsinMA_general', 'compulmsinMAniFA_general', 'infeccionresp_general', 'reagEPOC_general', 'fibrodiag_general', 'fibroterap_general', 'fistulapleu_general', 'FAdenovo_general', 'isqcard_general', 'infeccherida_general', 'IRA_general', 'otrasionfecc_general', 'exudado', 'eritema', 'exudadoPurulento', 'searacionTejidoProfundo', 'puntosAdicionalesTexto', 'otrasComplicaciones_0', 'otrasComplicaciones_1', 'otrasComplicaciones_2', 'otrasComplicaciones_7', 'otrasComplicaciones_30', 'clasif1cacionClaveDildo', 'URPA>3h', 'URPA>3h_NumHoras', 'URPA>3h_Causa', 'ingresoUci', 'fechaAltaUci', 'altaUCI_Vivo', 'reingresoUci', 'diasEstanciaReingreso', 'reingresoUCI_VMNI', 'reingresoUCI_VMI', 'reintervencion', 'transfusion', 'reingresoHospital', 'estadoAlAltaHospitalaria', 'altahosp_exitus', 'estado30DiasPostCirugia', 'estado180DiasPostCirugia', 'estado365DiasPostCirugia', 'Severe PPCs_0', 'Moderate PPCs_0', 'Severe PPCs_1', 'Moderate PPCs_1', 'Mild PPCS_1', 'Severe PPCs_2', 'Moderate PPCs_2', 'Severe PPCs_3', 'Moderate PPCs_3', 'Mild PPCS_3', 'Severe PPCs_5', 'Moderate PPCs_5', 'Mild PPCS_5', 'Severe PPCs_7', 'Moderate PPCs_7', 'Mild PPCS_7', 'Severe PPCs_30', 'Moderate PPCs_30', 'Severe_all', 'Severe_5', 'Moderate_all', 'Moderate_5', 'Mild_all', 'Mild_5', 'PPC_all']

df_clean = df.dropna(subset=columns_of_interest, thresh=5)

# Comprobación de normalidad
numeric_columns = ['edad', 'genero','días de estancia','altura', 'peso', 'IMC', 'ASA', 'aRISCAT', 'spO2pre(21)', 'ClinicalFrailityScale', 'hbPreoperatoria', 'indiceCHARLSON', 'apfel', 'pH', 'peepBasal', 'peepT1', 'peepT2', 'peepT3', 'peepMedia', 'd_peep', 'peepFinal', 'frBasal', 'frT1', 'frT2', 'frT3', 'frMedia', 'frFinal', 'vtBasal', 'vtT1', 'vtT2', 'vtT3', 'vtMedia', 'vtFinal', 'spo2Basal', 'spo2T1', 'spo2T2', 'spo2T3', 'spo2Media', 'spo2Final', 'fio2Basal', 'fio2T1', 'fio2T2', 'd_PaFi', 'fio2T3', 'd_SaFi', 'fio2Media', 'PaFi_Final', 'PaFI_Media', 'fio2Final', 'SaFi_Final', 'SaFi_Media', 'SaFi_Basal', 'PaFi_Basal', 'pao2Basal', 'pao2T1', 'pao2T2', 'pao2T3', 'pao2Media', 'pao2Final', 'pafiBasal', 'pafiMedia', 'pafiFinal', 'paco2Basal', 'paco2T1', 'paco2T2', 'paco2T3', 'paco2Media', 'paco2Final', 'pHBasal', 'pHT1', 'pHT2', 'pHT3', 'pHMedia', 'pHFinal', 'oriBasal', 'oriMedia', 'oriFinal', 'indicePerfusionBasal', 'indicePerfusionMedia', 'indicePerfusionFinal', 'pviBasal', 'pviMedia', 'pviFinal', 'sphbBasal', 'sphbMedia', 'sphbFinal', 'presionPicoBasal', 'presionPicoT1', 'presionPicoT2', 'presionPicoT3', 'presionPicoMedia', 'presionPicoFinal', 'presionMesetaBasal', 'presionMesetaT1', 'presionMesetaT2', 'presionMesetaT3', 'presionMesetaMedia', 'presionMesetaFinal', 'crsBasal', 'crsT1', 'crsT2', 'crsT3', 'crsMedia', 'pamBasal', 'pamT1', 'pamT2', 'pamT3', 'pamMedia', 'pamFinal', 'icBasal', 'icT1', 'icT2', 'icT3', 'icMedia', 'icFinal', 'vVSvPPBasal', 'vVSvPPMedia', 'vVSvPPFinal', 'tEsogagicaBasal', 'tEsogagicaMedia', 'tEsogagicaFinal', 'glucemiaBasal', 'glucemiaMedia', 'glucemiaFinal', 'volumen total', 'cristaloides', 'cHematies', 'coloides', 'diuresis', 'perdidaSangreEstimada', 'duracionCirugia', 'duracionVM', 'duracionOLV', 'duracionOLV_TLV', 'presionaperturamax', 'volumencorrienteunipulmonar', 'driving_pre_postMRA', 'numeroMRAnecesarias', 'numeroMRAfinalizadas', 'numeroMRAfracasadas', 'tiempoVMHastaExtubacion (minutos)', 'minutosTrasCirugia_15_eva', 'minutosTrasCirugia_15_do1sFarmaco', 'minutosTrasCirugia_60_eva', 'minutosTrasCirugia_120_eva', 'minutosTrasCirugia_180_eva', 'airTest_SpO2', 'airTest_PI', 'airTest_ORI', 'airTest_PVI', 'airTest_SpHb', 'tiempoDesdeEntradaURPA', 'gasometriaPostoperatoria_SpO2', 'gasometriaPostoperatoria_PaO2', 'gasometria_Post_FiO2', 'gasometriaPostoperatoria_PaO2_FIO2', 'gasometriaPostoperatoria_PaCO2', 'gasometriaPostoperatoria_pH', 'indiceNNIS', 'SOFA_0', 'SpO2_0', 'FIO2_0', 'PaO2_0', 'PaFi_0', 'PaCO2_0', 'pH_0', 'FrecResp_0', 'SOFA_1', 'SpO2_1', 'FIO2_1', 'PaO2_1', 'PaFi_1', 'PaCO2_1', 'pH_1', 'FrecResp_1', 'TAM_1', 'FC_1', 'SOFA_2', 'SpO2_2', 'FIO2_2', 'PaO2_2', 'PaFi_2', 'PaCO2_2', 'pH_2', 'FrecResp_2', 'SOFA_7', 'SpO2_7', 'FIO2_7', 'PaO2_7', 'PaFi_7', 'PaCO2_7', 'pH_7', 'FrecResp_7', 'URPA>3h_NumHoras', 'diasEstanciaReingreso']

#print("\nCantidad de registros por columna (sin valores NaN):")
#for columna in numeric_columns:
#    conteo = df[columna].dropna().shape[0]
#    print(f"{columna}: {conteo}")

normality_results = check_normality(df_clean, numeric_columns)

print("\nResultados del test de Shapiro-Wilk (comprobación de normalidad):")
print(normality_results)

with pd.ExcelWriter(output_excel, mode='w', engine='openpyxl') as writer:
    normality_results.to_excel(writer, sheet_name='Normalidad', index=False)
    #pd.DataFrame(contrast_results).to_excel(writer, sheet_name='Contrastes', index=False)

# Guardar gráficos de distribuciones
for col in numeric_columns:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_clean[col].dropna(), kde=True, bins=20)
    plt.title(f'Distribución de {col}')
    plt.xlabel(col)
    plt.ylabel('Frecuencia')
    file_path = os.path.join(output_dir, f'{col}_distribution_airtestdico_1_rescate_1.png')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()

# Contrastes de hipótesis
contrast_results = []

# Contraste Tipo de cirugía y SSI
#
# Contrastes de saturación crítica
#timepoints = ['SpO2_1', 'SpO2_2']
#for timepoint in timepoints:
#    df_clean['SpO2_crit'] = (df_clean[timepoint] < 92).astype(int)
#    contingency_table = pd.crosstab(df_clean['SpO2_crit'], df_clean['Composite infección'])
#    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
#    contrast_results.append({"Contraste": f"Saturación crítica (<92%) y CI en {timepoint}",
#                             "Chi2 Statistic": chi2_stat, "p-value": chi2_p})
#    print(f"\nAsociación entre saturación crítica (<92%) y CI en {timepoint} (Chi-cuadrado):")
#    print({"Chi2 Statistic": chi2_stat, "p-value": chi2_p})

# Guardar resultados en Excel

print(f"\nResultados guardados en: {output_excel}")
print(f"Gráficas guardadas en la carpeta: {output_dir}")
