#################################
Pastas
#################################
OriginalVideosUncut - Contém os videos originais da VSports da liga portugal 24/25
roboflow_sports_footar - repositório modificado do https://github.com/roboflow/sports
datasets_and_models - modelos criados através dos videos da primeira ronda da liga portugal 24/25, retirados do repositórios do roboflow e treinados localmente com o comando "yolo":
    - Jogadores, árbitros, guarda-redes, e bola - https://app.roboflow.com/footar/football-players-and-ball-5pzj6/
    - Linhas do Campo: https://universe.roboflow.com/footar/football-field-detection-f07vi-mrnfw



#################################
roboflow_sports_footar
#################################

Pastas
    data - contém os modelos yolo utilizados
    videos - contém os ficheiros de video da VSports cortados, para apenas mostrar o angulo de camara principal, cortando todo o tipo de closeups e repetições

    run_command_examples.txt - contém exemplos de comandos que foram testados ao longo do tempo com os vários parâmetros que o script main.py permite. Na alteração footar, permite passar pastas como argumentos, de input, e de output para processar multiplos ficheiros de uma só vez

    sports/common/team.py - alterado pela footar para deteção de equipa com K-means. O parametro DEBUG no inicio do ficheiro quando colocado a True, fará o debug passo a passo da extração das crops de jogadores das deteções de imagem, e mostra as crops, e os gráficos 3D de classificação por cores/equipa. Com o DEBUG ativo, pode-se clicar na tecla "q" para fazer skip a cada imagem/gráfico que é mostrado.
    -A deteção de cores foi feita com o formato "lab", em vez de RGB, devido a ser mais adequada a ser agrupada nos clusters por cores semelhantes num espaço 3 dimensões (L, A, B), do que em RGB que não tem tanto significado a nivel de semelhança de cores num espaço tridimensional.
    há muito lixo neste ficheiro de codigo comentado e tentativas falhadas de deteção de cor e classificação de equipas

main.py - o modelo original de deteção da bola não foi utilizado por se provar muito lento e por dividir a imagem em várias partes para análise. Foi utilizado com muito mais sucesso o modelo que já contém os jogadores,árbitros,guarda-redes e bola, por detetar a bola com bastante eficácia, e com muito melhor performance do que o modelo orignal.


#################################
Notas
#################################
O argumento de device deverá ser "cuda" se estiver a correr num computador com suporte a Nvidia com cuda cores disponíveis.
Em macOS M1 e derivados, deverá ser utilizado como device o "mps"
Caso nao exista gráfica dedicada, deverá ser utilizado como device "cpu", mas torna-se lento desta forma

