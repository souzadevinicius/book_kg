from nlp_utils import count_cooccurrence, count_cooccurrence2, text_analysis
import pandas as pd
from book_kg import book_analysis
chapter_content = """
Egito, Mesopotâmia e Creta 
 Alguma forma de arte existe em todas as regiões do globo, mas a história da arte como um esforço contínuo não começa nas cavernas do sul da França nem entre os índios norte-americanos. Não há uma tradição direta que ligue esses estranhos primórdios aos nossos dias, mas existe uma tradição direta, transmitida de mestre a discípulo, e de discípulo a admirador ou copista, a qual vincula a arte do nosso tempo, cada construção ou cada cartaz, à arte do vale do Nilo de uns cinco mil anos atrás. Pois iremos ver que os mestres gregos foram à escola com os egípcios, e todos nós somos discípulos dos gregos. Assim, a arte do Egito reveste-se de tremenda importância para nós. 
 Todos sabemos que o Egito é a terra das pirâmides ( Fig. 31 ), essas montanhas de pedra que se erguem no longínquo da história como marcos desgastados pelas intempéries. Por mais remotas e misteriosas que pareçam, elas nos revelam muito da sua história. Falam-nos de uma terra que estava tão perfeitamente organizada que foi capaz de empilhar esses gigantescos morros tumulares durante a vida de um único monarca, e falam-nos de reis que eram tão ricos e poderosos que puderam forçar milhares e milhares de trabalhadores ou escravos a labutar para eles, ano após ano, a cortar pedras nas canteiras, a arrastá-las ao local da construção e a deslocá-las com recursos sumamente primitivos até o túmulo ficar pronto para receber o faraó. Nenhum monarca e nenhum povo teria suportado semelhante gasto e passado por tantas dificuldades se se tratasse da criação de um mero monumento. Sabemos, porém, que as pirâmides tinham, de fato, importância prática aos olhos dos reis e seus súditos. O faraó era considerado um ser divino que exercia completo domínio sobre seu povo e que, ao partir deste mundo, voltava para junto dos deuses dos quais viera. As pirâmides, erguendo-se em direção ao céu, ajudá-lo-iam provavelmente a realizar essa ascensão. Em todo caso, elas preservariam seu corpo sagrado da decomposição. Pois os egípcios acreditavam que o corpo tinha que ser preservado a fim de que a alma pudesse continuar vivendo no além. Por isso impediam a desintegração do cadáver, graças a um elaborado método de embalsamar e enfaixar em tiras de pano. Era para a múmia do rei que a pirâmide fora erigida, e seu corpo ficava depositado justamente no centro da gigantesca montanha de pedra, num pétreo esquife. Em toda a volta da câmara funerária, eram escritos fórmulas mágicas e encantamentos para ajudá-lo em sua jornada para o outro mundo. 
 
 
 31 
 As pirâmides de Gizé, c.  2613-2563 a.C. 
 Mas não são apenas essas antiquíssimas relíquias da arquitetura humana que nos contam o papel desempenhado por vetustas crenças na história da arte. Os egípcios acreditavam que apenas preservar o corpo não era bastante, mas que, se uma fiel imagem do rei fosse preservada, não havia a menor dúvida de que ele continuaria vivendo para sempre. Assim, faziam com que artistas esculpissem a cabeça do rei em imperecível granito e a colocavam na tumba, onde ninguém a via, a fim de aí exercer sua magia e ajudar a alma a manter-se viva na imagem e através dela. Um nome egípcio para designar o escultor era, de fato, “Aquele que mantém vivo”. 
 Inicialmente, esses ritos eram reservados aos monarcas, mas logo os nobres da casa real passaram a ter seus túmulos menores agrupados em filas muito bem alinhadas ao redor do túmulo real; e gradualmente, toda pessoa que se prezava tinha que tomar providências para a vida no além, encomendando uma dispendiosa tumba para abrigar sua múmia e sua imagem, e onde sua alma podia habitar e receber as oferendas de alimento e bebida que eram feitas aos mortos. Alguns desses primeiros retratos da era das pirâmides, a quarta “dinastia” do “Antigo Império”, estão entre as mais belas obras da arte egípcia ( Fig. 32 ). Emanam desses retratos uma solenidade e simplicidade difíceis de esquecer. Vê-se que o escultor não estava tentando lisonjear o seu modelo nem preservar uma expressão fugidia. Interessava-se rigorosamente pelos aspectos essenciais. Ficavam excluídos todos os detalhes secundários. Talvez seja por causa dessa rigorosa concentração nas formas básicas da cabeça humana que esses retratos permanecem tão impressionantes. Pois, apesar da sua rigidez quase geométrica, não são tão primitivos quanto as máscaras indígenas de que nos ocupamos no Cap. 1 (pp. 47, 51,  Figs. 25 ,  28 ). Nem tão fiéis à realidade quanto os retratos naturalistas dos escultores nigerianos (p. 45,  Fig. 23 ). A observação da natureza e a regularidade do todo são equilibradas de um modo tão uniforme que essas cabeças nos impressionam por sua expressão de vida, sendo, no entanto, remotas e permanentes. 
 Essa combinação de regularidade geométrica e penetrante observação da natureza é característica de toda a arte egípcia. Podemos estudá-la melhor nos relevos e pinturas que adornavam as paredes dos túmulos. Destaquemos, contudo, que a palavra “adornar” ajusta-se mal a uma arte que devia ser vista apenas pela alma do morto. De fato, essas obras não tinham a finalidade de provocar deleite. A rigor elas se destinavam a “manter vivo”. Outrora, num passado sombrio e distante, era costume, quando morria um homem poderoso, que seus servos o acompanhassem na sepultura. Sacrificavam-nos para que o senhor chegasse ao além com um séquito condigno. Mais tarde esses horrores foram considerados cruéis, ou quiçá onerosos demais, e a arte acudiu para ajudar. Em vez de servidores de carne e osso, aos poderosos da Terra passaram a ser oferecidas imagens como substitutos. As pinturas e os modelos encontrados em túmulos egípcios estavam associados à ideia de fornecer servos para a alma no outro mundo, uma crença que é encontrada em muitas culturas antigas. 
 
 
 32 
 Cabeça, c.  2551-2528 a.C. 
 Encontrado num túmulo em Gizé; calcário, altura 27,8 cm; Kunsthistorisches Museum, Viena 
"""


import spacy

nlp = spacy.load('pt_core_news_sm')

# def test_cooccurence():
#     doc = nlp(chapter_content)
#     x = count_cooccurrence(doc)
#     print(x)

# def test_text_analysis():
#     occurrences, co_occurrences = text_analysis(chapter_content, model=nlp)
#     print(occurrences)



def test_book_analysis():
    book = pd.read_csv('./resources/historia_da_arte.csv')
    book = book.iloc[15:18]

    nlp = spacy.load('pt_core_news_sm')

    # Apply parse_chapter once and get both occurrences and co-occurrences
    analysed_text_df = book_analysis(book=book, excluded_words=["c", "el", "in", "i", "or", "di"], model=nlp)

    occurences_df = pd.concat(analysed_text_df["occurrences"].values)
    cooccurences_df = pd.concat(analysed_text_df["cooccurrences"].values)


    # sum_occurences_all_book(occurences_df).to_csv("occurences.csv", index = None)
    # sum_occurences_all_book(cooccurences_df).to_csv("cooccurences.csv", index = None)
    occurences_df.to_csv("occurences.csv", index = None)
    cooccurences_df.to_csv("cooccurences.csv", index = None)
    