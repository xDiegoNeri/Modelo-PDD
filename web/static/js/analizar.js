/**
 * Script para la página de análisis de imágenes de plantas
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elementos del DOM
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const changeImageBtn = document.getElementById('change-image');
    const analyzeBtn = document.getElementById('analyze-btn');
    const uploadForm = document.getElementById('upload-form');
    const loadingIndicator = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    
    // Prevenir comportamiento por defecto para eventos de arrastrar y soltar
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Resaltar área de soltar cuando se arrastra un archivo sobre ella
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        dropArea.classList.add('highlight');
    }
    
    function unhighlight() {
        dropArea.classList.remove('highlight');
    }
    
    // Manejar archivos soltados
    dropArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }
    
    // Manejar selección de archivo mediante el input
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files);
        }
    });
    
    // Procesar archivos seleccionados
    function handleFiles(files) {
        const file = files[0]; // Solo procesamos el primer archivo
        
        // Verificar que sea una imagen
        if (!file.type.match('image.*')) {
            alert('Por favor, selecciona una imagen válida (JPEG, PNG, etc.)');
            return;
        }
        
        // Mostrar vista previa
        previewFile(file);
    }
    
    // Mostrar vista previa de la imagen
    function previewFile(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
            dropArea.style.display = 'none';
        };
        
        reader.readAsDataURL(file);
    }
    
    // Cambiar imagen
    changeImageBtn.addEventListener('click', function() {
        previewContainer.style.display = 'none';
        dropArea.style.display = 'block';
        fileInput.value = ''; // Limpiar input para permitir seleccionar el mismo archivo
    });
    
    // Manejar envío del formulario
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Verificar que se haya seleccionado un archivo
        if (!fileInput.files.length) {
            alert('Por favor, selecciona una imagen para analizar.');
            return;
        }
        
        // Mostrar indicador de carga
        previewContainer.style.display = 'none';
        loadingIndicator.style.display = 'block';
        
        // Crear FormData y enviar
        const formData = new FormData(this);
        
        fetch(this.action, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Error en la respuesta del servidor');
            }
            return response.json();
        })
        .then(data => {
            // Ocultar indicador de carga
            loadingIndicator.style.display = 'none';
            
            // Mostrar resultados
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            loadingIndicator.style.display = 'none';
            alert('Ha ocurrido un error al procesar la imagen. Por favor, intenta de nuevo.');
            previewContainer.style.display = 'block';
        });
    });
    
    // Mostrar resultados de la predicción
    function displayResults(data) {
        if (!data.success) {
            alert('Error: ' + data.error);
            previewContainer.style.display = 'block';
            return;
        }
        
        // Configurar imagen de resultado
        document.getElementById('result-image').src = previewImage.src;
        
        // Mostrar predicción principal
        document.getElementById('prediction-result').textContent = data.class_es;
        document.getElementById('confidence').textContent = `Confianza: ${data.confidence.toFixed(2)}%`;
        
        // Generar gráfico de barras para las predicciones
        const barChart = document.getElementById('bar-chart');
        barChart.innerHTML = ''; // Limpiar contenido previo
        
        // Ordenar predicciones por confianza (de mayor a menor)
        const predictions = data.top_predictions.sort((a, b) => b.confidence - a.confidence);
        
        // Mostrar las 5 principales predicciones
        predictions.forEach(pred => {
            const barItem = document.createElement('div');
            barItem.className = 'bar-item';
            
            const barLabel = document.createElement('div');
            barLabel.className = 'bar-label';
            
            const barName = document.createElement('span');
            barName.className = 'bar-name';
            barName.textContent = pred.class_es;
            
            const barValue = document.createElement('span');
            barValue.className = 'bar-value';
            barValue.textContent = `${pred.confidence.toFixed(1)}%`;
            
            barLabel.appendChild(barName);
            barLabel.appendChild(barValue);
            
            const barContainer = document.createElement('div');
            barContainer.className = 'bar-container';
            
            const bar = document.createElement('div');
            bar.className = 'bar';
            bar.style.width = `${pred.confidence}%`;
            
            barContainer.appendChild(bar);
            
            barItem.appendChild(barLabel);
            barItem.appendChild(barContainer);
            
            barChart.appendChild(barItem);
        });
        
        // Mostrar información sobre la enfermedad
        const diseaseInfo = getDiseaseInfo(data.class);
        if (diseaseInfo) {
            document.getElementById('disease-description').innerHTML = `<p>${diseaseInfo.description}</p>`;
            
            const treatmentList = document.createElement('ul');
            diseaseInfo.treatment.forEach(item => {
                const li = document.createElement('li');
                li.textContent = item;
                treatmentList.appendChild(li);
            });
            
            document.getElementById('treatment-recommendations').innerHTML = '';
            document.getElementById('treatment-recommendations').appendChild(treatmentList);
        }
        
        // Mostrar sección de resultados
        resultsSection.style.display = 'block';
        
        // Desplazar a la sección de resultados
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Información sobre enfermedades (podría cargarse desde el servidor en una aplicación real)
    function getDiseaseInfo(diseaseName) {
        const diseaseInfo = {
            'Blight': {
                description: 'El tizón es una enfermedad fúngica común que afecta a diversas plantas, causando la muerte rápida del tejido vegetal. Existen varios tipos, como el tizón temprano y el tizón tardío, que afectan principalmente a cultivos como tomates, papas y otras solanáceas.',
                treatment: [
                    'Aplicar fungicidas a base de cobre o productos específicos para tizón',
                    'Eliminar y destruir las partes afectadas de la planta',
                    'Mejorar la circulación de aire entre las plantas',
                    'Evitar regar el follaje; regar directamente en la base',
                    'Rotar cultivos para prevenir la acumulación del patógeno en el suelo'
                ]
            },
            'Curl': {
                description: 'El enrollamiento de hojas es un síntoma que puede ser causado por virus, ácaros o factores ambientales. El virus del enrollamiento de la hoja es particularmente común en cultivos como tomate, pimiento y papas.',
                treatment: [
                    'Controlar insectos vectores como mosca blanca y pulgones',
                    'Eliminar plantas infectadas para prevenir propagación',
                    'Utilizar variedades resistentes cuando sea posible',
                    'Mantener buenas prácticas de higiene en el jardín o cultivo',
                    'No existe cura para plantas infectadas con virus'
                ]
            },
            'Green Mite': {
                description: 'Los ácaros verdes son pequeñas arañas que se alimentan de la savia de las plantas. Son difíciles de ver a simple vista pero causan daños significativos al debilitar la planta y reducir su capacidad fotosintética.',
                treatment: [
                    'Aplicar acaricidas específicos siguiendo las instrucciones',
                    'Utilizar aceites hortícolas o jabones insecticidas',
                    'Introducir depredadores naturales como ácaros predadores',
                    'Aumentar la humedad ambiental (los ácaros prefieren condiciones secas)',
                    'Lavar las hojas con agua a presión para eliminar ácaros'
                ]
            },
            'Healthy': {
                description: 'Tu planta parece estar saludable. No se detectan signos de enfermedades o plagas en la imagen analizada.',
                treatment: [
                    'Continúa con los cuidados regulares de riego y fertilización',
                    'Mantén una buena circulación de aire alrededor de la planta',
                    'Realiza inspecciones periódicas para detectar problemas temprano',
                    'Aplica fertilizantes según las necesidades específicas de la planta',
                    'Protege la planta de condiciones climáticas extremas'
                ]
            },
            'Leaf Miner': {
                description: 'Los minadores de hojas son larvas de insectos que viven dentro del tejido de las hojas, creando túneles o galerías mientras se alimentan. Afectan a una amplia variedad de plantas ornamentales y cultivos.',
                treatment: [
                    'Eliminar y destruir hojas afectadas',
                    'Aplicar insecticidas sistémicos que penetren en el tejido foliar',
                    'Utilizar trampas amarillas pegajosas para capturar adultos',
                    'Introducir avispas parasitoides como control biológico',
                    'Mantener las plantas sanas y vigorosas para resistir el daño'
                ]
            },
            'Leaf Spot': {
                description: 'Las manchas foliares son enfermedades causadas principalmente por hongos, aunque también pueden ser bacterianas. Se caracterizan por lesiones localizadas en las hojas que pueden variar en tamaño, forma y color según el patógeno.',
                treatment: [
                    'Aplicar fungicidas preventivos o curativos según el caso',
                    'Eliminar y destruir hojas infectadas',
                    'Evitar regar por aspersión; regar en la base de la planta',
                    'Mejorar la circulación de aire entre plantas',
                    'Rotar cultivos para reducir la presencia del patógeno'
                ]
            },
            'Mosaic': {
                description: 'El mosaico es una enfermedad viral que afecta a numerosas especies de plantas. El virus del mosaico del tabaco (TMV) es uno de los más comunes, pero existen muchas variantes específicas para diferentes cultivos.',
                treatment: [
                    'No existe cura para plantas infectadas con virus',
                    'Eliminar y destruir plantas infectadas',
                    'Controlar insectos vectores como pulgones y trips',
                    'Desinfectar herramientas de jardinería entre plantas',
                    'Utilizar variedades resistentes cuando estén disponibles'
                ]
            },
            'Powdery': {
                description: 'El oídio o mildiu polvoriento es una enfermedad fúngica común que afecta a una amplia gama de plantas. Se desarrolla principalmente en condiciones de alta humedad pero con hojas secas, y temperaturas moderadas.',
                treatment: [
                    'Aplicar fungicidas específicos para oídio',
                    'Utilizar remedios caseros como solución de bicarbonato de sodio',
                    'Mejorar la circulación de aire entre plantas',
                    'Evitar exceso de fertilización nitrogenada',
                    'Podar partes afectadas para reducir la propagación'
                ]
            },
            'Rust': {
                description: 'La roya es una enfermedad fúngica caracterizada por pústulas de color óxido o marrón rojizo. Existen muchos tipos de roya, cada uno específico para ciertas plantas o grupos de plantas relacionadas.',
                treatment: [
                    'Aplicar fungicidas preventivos al inicio de la temporada',
                    'Utilizar fungicidas sistémicos para infecciones establecidas',
                    'Eliminar y destruir partes afectadas',
                    'Evitar mojar el follaje al regar',
                    'Mantener buena circulación de aire entre plantas'
                ]
            },
            'Streak Virus': {
                description: 'El virus del rayado afecta principalmente a cereales y gramíneas, aunque existen variantes que afectan a otras plantas. Se transmite principalmente por insectos vectores y puede causar pérdidas significativas en cultivos.',
                treatment: [
                    'No existe cura para plantas infectadas con virus',
                    'Eliminar y destruir plantas infectadas',
                    'Controlar insectos vectores como saltahojas y pulgones',
                    'Utilizar variedades resistentes cuando estén disponibles',
                    'Implementar rotación de cultivos y prácticas culturales adecuadas'
                ]
            }
        };
        
        return diseaseInfo[diseaseName];
    }
});