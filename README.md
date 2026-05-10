Como Executar

1. Instalar dependências<br>
bashpip install -r requirements.txt

2. Definir a chave da API Anthropic<br>
bashexport ANTHROPIC_API_KEY="sk-ant-..."

3. Rodar o pipeline<br>
bashpython rag.py

## HNSW e Consumo de Memória

O índice HNSW (Hierarchical Navigable Small World) utiliza estruturas de grafos para acelerar a busca vetorial aproximada. Diferente do KNN exato, que compara a consulta com todos os vetores armazenados, o HNSW cria conexões entre vetores vizinhos para reduzir o número de comparações durante a busca.

Os principais hiperparâmetros utilizados foram:

`M`: define a quantidade de conexões entre os nós do grafo. Valores maiores aumentam a precisão da busca, mas também elevam o consumo de memória RAM, pois mais conexões precisam ser armazenadas.

`ef_construction`: define quantos candidatos são analisados durante a construção do índice. Valores maiores produzem um grafo mais preciso, porém aumentam o tempo de indexação e o uso de memória durante a construção.

Em comparação ao KNN exato, o HNSW consome mais memória devido à estrutura adicional do grafo, mas oferece buscas muito mais rápidas e escaláveis para grandes volumes de embeddings.