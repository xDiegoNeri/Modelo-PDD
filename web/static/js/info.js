/**
 * Script para la página de información sobre enfermedades
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elementos del DOM
    const filterButtons = document.querySelectorAll('.filter-btn');
    const diseaseCards = document.querySelectorAll('.disease-info-card');
    
    // Añadir evento de clic a los botones de filtro
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Remover clase active de todos los botones
            filterButtons.forEach(btn => btn.classList.remove('active'));
            
            // Añadir clase active al botón clickeado
            this.classList.add('active');
            
            // Obtener categoría a filtrar
            const filterValue = this.getAttribute('data-filter');
            
            // Filtrar tarjetas
            filterDiseaseCards(filterValue);
        });
    });
    
    // Función para filtrar tarjetas de enfermedades
    function filterDiseaseCards(category) {
        diseaseCards.forEach(card => {
            if (category === 'all') {
                // Mostrar todas las tarjetas
                card.style.display = 'block';
                setTimeout(() => {
                    card.style.opacity = '1';
                }, 50);
            } else {
                // Filtrar por categoría
                const cardCategory = card.getAttribute('data-category');
                
                if (cardCategory === category) {
                    // Mostrar tarjeta con animación
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                    }, 50);
                } else {
                    // Ocultar tarjeta con animación
                    card.style.opacity = '0';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            }
        });
    }
    
    // Inicializar con todas las tarjetas visibles
    filterDiseaseCards('all');
});