from keras.models import model_from_json
from keras.models import model_from_json

import pickle
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 700
def obtain_classes(proba, multilabel_binarizer):
    idxs = np.argsort(proba)[::-1][:10]
    # loop over the indexes of the high confidence class labels
    for (i, j) in enumerate(idxs):
        if (proba[j] * 100 > 0.5):
            # build the label and draw the label on the image
            label = "{}: {:.2f}%".format(multilabel_binarizer.classes_[j], proba[j] * 100)
            print(label)
            

def main():
    # loading
    with open('../data/neural_network_config/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load json and create model
    json_file = open('../data/neural_network_config/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    # load weights into new model
    loaded_model.load_weights("../data/neural_network_config/temp-model-new.h5")
    print("Loaded model from disk")

    # load categories
    with open('../data/neural_network_config/multilabel_binarizer.pickle', 'rb') as f:
        multilabel_binarizer = pickle.load(f)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])



    sentence_test = ["La entrada en el 2020 ha comportado cambios en la edad de jubilación y en el cálculo de los años cotizados que se tienen en cuenta para determinar la prestación. Las medidas son de carácter automático, ya que forman parte de la reforma de 2011 que hace que la edad para jubilarse se retrase paulatinamente hasta llegar a los 67 años.  ADVERTISING  Todo esto se da mientras resta pendiente saber cuál será la subida de las prestaciones en el 2020, ya que aunque el Gobierno en funciones ha prometido que se subirán el 0,9% y no perderán poder adquisitivo, la medida no se tomará hasta que esté formado un Ejecutivo. En diciembre de 2019 en España se contabilizaban 6.089.294 pensiones de jubilación, con una prestación media de 1.143,55 euros mensuales.   Pensiones en 2020 Los cambios en la edad de jubilación  Respecto a la edad de jubilación, cada año se va retrasando en virtud del régimen establecido en la reforma de 2011 aprobada durante el mandato de José Luis Rodríguez Zapatero. De esta forma, en 2020 la edad legal ordinaria será de 65 años y 10 meses. Esta edad se aplicará a aquellos que han cotizado menos de 37 años.  Si una persona llega a los 65 años en 2020 y ha cotizado 37 años o más, ya podrá jubilarse con 65 años.  En el caso de la jubilación parcial, en la que se combina trabajo y prestación, el mínimo será de 61 años y 10 meses con 35 años o más cotizados; o de 62 años y 8 meses con 33 años cotizados.  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia Pensiones en 2020 Los cambios en el cálculo de la pensión  Por lo que respecta al cálculo de la pensión que se cobrará la momento de jubilarse, en 2020 se tendrán en cuenta los últimos 23 años cotizados. Estos años cotizados conforman la base reguladora, que es la suma de las bases de cotización en dicho periodo. Hay que tener en cuenta que cuantos más años se tengan en cuenta es posible que se recorte más la pensión, ya que en los últimos años de vida laboral es cuando mejores salarios se suelen cobrar.   Esta es otra de las reformas introducidas con los cambios en las pensiones de la década anterior, momento hasta el que se tenían en cuenta los últimos 15 años trabajados. La idea es que para 2022 ya se tengan en cuenta los últimos 25 años cotizados. De esta manera, en 2021 se computarán los últimos 24 años trabajados y en 2022 los últimos 25 años cotizados.  La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente (LV) En 2023 El recorte de las pensiones que viene  Otra de las medidas que tendrán un fuerte calado en el sistema es la llegada del factor de sostenibilidad, que se aplicará a partir de 2023 e irá recortando las nuevas pensiones, teniendo en cuenta que los pensionistas vivirán más. Dicha medida en un principio debía aplicarse en 2019.  El conjunto de medidas se puede consultar al detalle en la guía para la jubilación del Ministerio de Trabajo, Migraciones y Seguridad Social.",
    "PSOE y PNV pactan “adecuar el Estado” para reconocer las identidades territoriales  7El pacto de investidura con los jeltzales, acelerado por la negociación con ERC, aboga por afrontar “las modificaciones legales necesarias” para dar “solución al contencioso en Catalunya” PSOE y PNV pactan “adecuar el Estado” para reconocer las identidades territoriales Sánchez y Ortuzar firman un acuerdo para la investidura (Emilia Gutiérrez) PEDRO VALLÍN, MADRID 30/12/2019 14:08 Actualizado a 30/12/2019 15:42 El presidente del Euskadi Buru Batzar, Andoni Ortuzar, no disimula su satisfacción: “Estamos contentos por poder aportar nuestra humilde contribución para sacar a la política de la parálisis en la que se encuentra”. El líder del PNV, tras firmar el acuerdo con el presidente en funciones, Pedro Sánchez, subrayaba que el acuerdo suscrito por ambos es una apuesta por el diálogo y por la desjudicialización de la política. “Este acuerdo es un ejercicio de responsabilidad para volver a poner a la política en su sitio, favorecer el diálogo entre diferentes y traer la máxima estabilidad”, explicó Ortuzar.  En cuanto a la fecha, el presidente del PNV indicó que por su formación no hay problema para la celebración de una sesión de investidura cuanto antes, incluidas fechas a priori festivas como el 5 de enero, domingo previo a la festividad de los Reyes Magos, pero no quiso prejuzgar cuál sería la fecha, toda ves, subrayó, faltan por pronunciarse otras “formaciones tan determinantes o más que el PNV” para ese acuerdo de investidura, en alusión a ERC.   El pacto del PNV con el PSOE incluye la interlocución prioritaria de la acción de gobierno, por la que el grupo vasco quiere ser informado a priori de aquellas medidas que el futuro ejecutivo vaya a adoptar y que afecten al país vasco, muy en particular, las que atañen a reformas fiscales.  La foto de Sánchez y Ortuzar supone un espaldarazo a la negociación de la mayoría necesaria para la investidura en un momento en que el pacto con ERC está pendiente de la aprobación definitiva por los órganos del partido, tras la publicación del escrito de Abogacía del Estado respecto a la situación legal de Oriol Junqueras.  Los doce puntos del pacto suscrito por PSOE-PNV, donde se alude tanto a la agenda vasca como al nuevo estatuto de autonomía, e incluso a la llamada “ley mordaza digital”, son los siguientes:  1.Mantener una comunicación fluida y constante con EAJ-PNV, dando a conocer con antelación suficiente los proyectos e iniciativas que el Gobierno desee impulsar, comprometiéndose, además, a llegar a un acuerdo satisfactorio en caso de discrepancia.  2.Mantener una comunicación fluida con el Gobierno vasco en aras de evitar la judicialización de las discrepancias, que debe ser sustituida por el acuerdo político.  3.Proceder en 2020 a la negociación y traspaso a la CAV de las competencias estatutarias pendientes. Así mismo, se procederá en el plazo de seis meses al traspaso de las competencias de tráfico a la Comunidad Foral de Navarra, con el mismo contenido y extensión que las realizadas en su momento a la CAV.   4.Impulsar, a través del diálogo entre partidos e instituciones, las reformas necesarias para adecuar la estructura del Estado al reconocimiento de las identidades territoriales, acordando, en su caso, las modificaciones legales necesarias, a fin de encontrar una solución tanto al contencioso en Catalunya como en la negociación y acuerdo del nuevo Estatuto de la CAV, atendiendo a los sentimientos nacionales de pertenencia.  5.Apostar de manera urgente, firme y decidida por las infraestructuras correspondientes al Estado en la CAV y, especialmente, por lo relativo al TAV, incluido el cronograma de trabajos y los compromisos complementarios ya cerrados con el Gobierno del Estado.  6.Cumplir con los aspectos pendientes de los acuerdos suscritos por EAJ-PNV con el Gobierno del PP en diferentes ámbitos, compromiso ya adquirido por el candidato en el trámite de la moción de censura de 2018.  7.Impulsar la construcción europea y la presencia y participación de las instituciones vascas en las instituciones de la Unión.  8.Impulsar decididamente la industria y compensar su adecuación a la transformación energética con el mantenimiento de la actividad económica y el empleo, posibilitando una transición realista y protegiendo los puestos de trabajo de los sectores afectados por estos cambios.  9.Acordar previamente con EAJ-PNV las medidas fiscales que el Gobierno quiera proponer a las Cortes, así como encauzar las discrepancias que puedan producirse en las relaciones en el ámbito fiscal o el financiero establecidas por el Concierto Económico.   10.El proceso de determinación de los objetivos de déficit correspondientes a la CAV y a la CFN, así como el de los criterios de reinversión del superávit de las entidades locales, diputaciones forales y gobiernos en sus respectivos territorios se realizará con la participación y en el marco de las Comisiones Mixtas de Concierto y Convenio.  11.Abrir cauces para promover la representación internacional de Euskadi en el ámbito deportivo y cultural.  12.Modificar, con el acuerdo de EAJ-PNV, el contenido de los conocidos como decretos digitales de manera que sean resueltas las discrepancias sobre los mismos manifestadas en el ámbito parlamentario (Real Decreto-ley 14/2019, de 31 de octubre y Real Decreto-ley 12/2018, de 7 de septiembre).",
    "Ni 24 horas. Es lo que ha durado el consenso que la ministra de Educación y Formación Profesional, Isabel Celaá, decía haber alcanzado con las comunidades autónomas para poner fin al año académico 2019/2020. Autonomías como Madrid, Andalucía, Murcia y País Vasco se han descolgado del acuerdo alcanzado este miércoles por la Conferencia Sectorial de Educación. Los Gobiernos del PP madrileño, andaluz y murciano no apoyan la promoción general de los alumnos. El PNV, por su parte, argumenta que tiene un propio plan para la finalización del curso en Euskadi. Así, los Ejecutivos autonómicos de Madrid y Murcia, liderados por los populares Isabel Díaz Ayuso y Fernando López Miras, respectivamente, no sólo no están de acuerdo con que los estudiantes puedan pasar de curso con carácter general. También han criticado que los alumnos de 4º de la ESO y 2º de Bachilleraro puedan obtener su título sin superar todas las materias. No se puede compartir la propuesta que establece, ni más ni menos, que se titule con asignaturas suspensas, argumentan. Es, por estos dos puntos en concreto, por los que no suscribirán el Acuerdo para el desarrollo del tercer trimestre del curso 2019/2020 y el inicio del curso 2020/2021. Sería injusto, un menosprecio al esfuerzo de los alumnos y una falta de respeto al enorme trabajo que están haciendo los docentes, explica, por su parte, Javier Imbroda, consejero de Educación de Andalucía, quien también se niega a firmar el plan de Celaá. Además, estos Gobiernos regionales del PP coinciden en que sería caótico que cada comunidad tomase diferentes decisiones en ese ámbito que pudieran generar desigualdades entre los alumnos, según comenta la consejera murciana, Esperanza Moreno. La ministra de Educación, Isabel Celaá, este miércoles en rueda de prensa. La ministra de Educación, Isabel Celaá, este miércoles en rueda de prensa. El PNV, pese a ver la intención del Ministerio de Educación con buenos ojos, tampoco rubricará el pacto porque son sólo orientaciones y el Gobierno vasco ya tiene una hoja de ruta propia para cerrar el año académico 2019/2020, haciendo uso de sus competencias en Educación. El 'aprobado para todos' de Celaá que permite titular con suspensos: dos meses sin 'tocar' los libros Joaquín Vera En tiempos de coronavirus, la tercera evaluación será un mero diagnóstico 'evaluado' siempre de manera positiva: repetir será la excepción. El acuerdo de Celáa Estas reacciones son consecuencia de que la ministra de Educación, Isabel Celáa, acordó con las comunidades autónomas que este curso académico no se extendería más allá de junio. Según ella, este miércoles en la Conferencia Sectorial, el máximo órgano de interlocución entre el Gobierno central y las regiones autonómicas en materia educativa, decidieron que todos los alumnos pasen de curso, pero no con la misma nota, fruto de la situación actual por la crisis del coronavirus. Y es que la manera de calificar a los alumnos depende, según este plan, del equipo docente de cada alumno. Así, los profesores de cualquier centro educativo que, en palabras de Celaá, son los que mejor conocen las aptitudes de sus alumnos, decidirán las calificaciones finales de cada estudiante basándose en las notas obtenidas por éste durante la primera y la segunda evaluación. No obstante, los alumnos seguirán cursando el tercere trimestre, que tendrá unas funciones especiales, como la de diagnosticar qué contenidos no quedan bien fijados para darlos el año que viene. Un alumna del colegio Estudio, realizando deberes. Un alumna del colegio Estudio, realizando deberes. La manera de evaluar se tomaría en una decisión colegiada en el marco que regulen las administraciones públicas autonómicas. Por ello, los consejeros de Educación no terminan de ver si sería justo que cada autonomía decida, mediante su propia legislación, cómo será este marco. Hemos pedido que se clarifiquen apartados para que no haya agravios con el resto de comunidades ni se generen desigualdades, ha detallado el consejero de Andalucía. La repetición de alumnos En cuanto a la decisión de que un estudiante repita el curso o no, desde el Ministerio de Educación confían plenamente en que los docentes califiquen a sus alumnos porque son los que mejor conocen sus capacidades. Pese a ello, siempre se va a intentar que el estudiante promocione, ya que las repeticiones serán muy excepcionales y, si fuera el caso, deberán estar sólidamente argumentadas y acompañadas con un plan de refuerzo. De hecho, Celaá ha insistido en la idea de que habrá actividades de evaluación que permitan que los alumnos que no hayan tenido notas satisfactorias en el primer y el segundo trimestre pueden mejorarlas en este tercero para poder promocionar. La disparidad que se pueda generar entre las comunidades, según argumentan los consejeros de Madrid, Andalucía, Murcia, en los criterios de decisión para que los profesores decidan si un alumno promociona o no es una injustica. Esto les empuja a no suscribir el acuerdo. Además, Galicia y Castilla y León -gobernadas por el PP- también se han sumado a estas quejas pero, por el momento, no se han descolgado del plan de Celaá.",
    "Arthur estará tres semanas más de baja y Ter Stegen es duda para el derbi  0El brasileño regresa de vacaciones con los mismos problemas en el pubis; el alemán arrastra molestias en una rodillaConsulta el calendario de la Liga Santander Arthur estará tres semanas más de baja y Ter Stegen es duda para el derbi Arthur jugó su último partido con el Barça el pasado 1 de diciembre (PIERRE-PHILIPPE MARCOU / AFP) REDACCIÓN 30/12/2019 13:46 Actualizado a 30/12/2019 17:44 El Barcelona ha vuelto a los entrenamientos tras las vacaciones navideñas con contratiempos. Los problemas de Arthur Melo en el pubis continúan. El centrocampista brasileño seguirá de baja al menos tres semanas más para tratar de resolver sus molestias, tal como ha informado el club en el comunicado emitido este lunes. No es el único jugador en la enfermería blaugrana, ya que en el mismo texto el Barça ha dado a conocer la lesión de Marc André Ter Stegen.  ADVERTISING  El portero alemán, padre por primera vez este domingo, está recibiendo un tratamiento en su rodilla derecha a causa de una tendinopatía, un problema físico que arrastra desde el último partido de Liga contra el Deportivo Alavés. Su presencia en el derbi barcelonés contra el Espanyol del próximo fin de semana queda en el aire. La evolución marcará su disponibilidad, reza el comunicado.   Neto será titular contra el Espanyol si el Barça prefiere no arriesgar con Ter Stegen  El grueso de la plantilla ya está a las órdenes de Ernesto Valverde tras unos días de descanso salvo os sudamericanos (Leo Messi, Luis Suárez y Arturo Vidal), con permiso del club hasta el 2 de enero, dos días antes del enfrentamiento entre el líder y el colista de la Liga. La excepción es Neto, quien será titular si se decide no arriesgar con Ter Stegen.  El caso de Arthur sigue siendo cuando menos intrigante. Jugó su último partido el pasado 1 de diciembre contra el Atlético de Madrid y desde entonces sus molestias en el pubis se han eternizado. Sin opciones de regresar a los terrenos de juego hasta finales de mes, se perderá como mínimo la Supercopa de España en Arabia Saudí y dos partidos de Liga (Espanyol y Granada). En lo que va de temporada ha jugado 12 partidos, marcando dos goles y repartiendo cuatro asistencias."]
    xnew = tokenizer.texts_to_sequences(sentence_test)
    xnew = pad_sequences(xnew, padding='post', maxlen=maxlen)
    print(xnew)

    ynew = loaded_model.predict(xnew)
    index = 0
    for proba in ynew:
        print("----")
        print(sentence_test[index])
        obtain_classes(proba, multilabel_binarizer)
        index = index +1


    #print(ynew)
    #print(multilabel_binarizer.inverse_transform(np.array([ynew])))

if __name__== "__main__":
  main()