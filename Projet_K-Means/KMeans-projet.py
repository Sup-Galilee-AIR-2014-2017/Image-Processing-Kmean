import cv2
import numpy
import random


def SetBarycentreAuxGraines(groupe, bleu, vert, rouge, cluster_bleu, cluster_rouge, cluster_vert, K, nb_pixels):
    # On parcourt chaque cluster (chaque graine)
    for i in range(0, K):
        #La formule du barycentre, pour chaque coordonnee (on prend ici x, Xn etant l'ensemble des x de cardinal vallant n), vaut Bar(Xn) = (Somme des x) / n
        nombreElementsDansCluster = 0;

        somme_bleu = 0
        somme_rouge = 0
        somme_vert = 0

        for pixel in range(0, nb_pixels):
            if(groupe[pixel, 0] == i):
                nombreElementsDansCluster += 1
                somme_bleu += bleu[pixel, 0]
                somme_rouge += rouge[pixel, 0]
                somme_vert += vert[pixel, 0]

        cluster_bleu[i] = somme_bleu / nombreElementsDansCluster;
        cluster_rouge[i] = somme_rouge / nombreElementsDansCluster;
        cluster_vert[i] = somme_vert / nombreElementsDansCluster;

def CalculeModule(debut_vert, debut_rouge, debut_bleu, fin_vert, fin_rouge, fin_bleu):

    te1 = numpy.power(fin_vert - debut_vert, 2)
    te2 = numpy.power(fin_rouge - debut_rouge, 2)
    te3 = numpy.power(fin_bleu - debut_bleu, 2)

    return numpy.sqrt( numpy.power(fin_vert - debut_vert, 2) + numpy.power(fin_rouge - debut_rouge, 2) + numpy.power(fin_bleu - debut_bleu, 2) )

def AttributionKMeans(groupe, bleu, rouge, vert, cluster_bleu , cluster_rouge, cluster_vert, K, nb_pixels):
    for pixel in range(0, nb_pixels):
        distance_graine = numpy.zeros(K)
        for graine in range(0, K):
            distance_graine[graine] = CalculeModule(vert[pixel,0], rouge[pixel,0], bleu[pixel,0], cluster_vert[graine], cluster_rouge[graine], cluster_bleu[graine])

        graineMinimale = 0
        distance_min = distance_graine[0]
        for graine in range(1, K):
            if(distance_graine[graine] < distance_min):
                distance_min = distance_graine[graine]
                graineMinimale = graine

        groupe[pixel] = graineMinimale


def main():
    MAX_LARGEUR = 400
    MAX_HAUTEUR = 400

    K = 32 #Le fameux parametre K de l'algorithme



    # Charger l'image et la reduire si trop grande (sinon, on risque de passer trop de temps sur le calcul...)
    imagecolor = cv2.imread('perr.jpg')
    if imagecolor.shape[0] > MAX_LARGEUR or imagecolor.shape[1] > MAX_HAUTEUR:
        factor1 = float(MAX_LARGEUR) / imagecolor.shape[0]
        factor2 = float(MAX_HAUTEUR) / imagecolor.shape[1]
        factor = min(factor1, factor2)
        imagecolor = cv2.resize(imagecolor, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)



    # Le nombre de pixels de l'image
    nb_pixels = imagecolor.shape[0] * imagecolor.shape[1]


    # On affiche une fenetre avec l'image
    # cv2.namedWindow("image")
    #On sort quand l'utilisateur appuie sur une touche
    cv2.imshow("image", imagecolor)
    key = cv2.waitKey(0)


    #Les coordonnees BRV de tous les pixels de l'image (les elements de E)
    bleu = imagecolor[:, :, 0].reshape(nb_pixels, 1)
    vert = imagecolor[:, :, 1].reshape(nb_pixels, 1)
    rouge = imagecolor[:, :, 2].reshape(nb_pixels, 1)




    #Les coordonnees BRV de chaque point-cluster (chaque graines = les elements de N)
    cluster_bleu = numpy.zeros(K)
    cluster_vert = numpy.zeros(K)
    cluster_rouge = numpy.zeros(K)


    #Ce tableau permet de connaitre, pour chaque pixel de l'image, a quel cluster il appartient
    #On le remplit au hasard
    groupe = numpy.zeros((nb_pixels, 1)) #groupe est un tableau de Card(E) cases, et chaque valeur est un entier entre 0 et K-1, designant le cluster auquel chaque point sera rattache
    #On remplit au hasard le tableau groupe, c'est a dire que l'on attribue au hasard chaque point de l'espace a un des K clusters
    #Cependant, pour etre sur qu'au depart chaque cluster est rattache a au moins un point de l'espace, on attribue les K premiers points de l'espace a chaque K clusters
    for i in range(0,K):
        groupe[i,0]=i
    #La, on fait l'attribution du reste des points de l'espace a des clusters choisis au hasard
    for i in range(K,nb_pixels):
        groupe[i,0]=random.randint(0, K-1)



        #########################
    ####### Debut code eleves #######
        #########################

    distanceReferenceDiagonaleCoordonneesCouleurs = CalculeModule(0,0, 0, 255, 255, 255) # numpy.sqrt(3 * numpy.power(255, 2)) # Reference par rapport a laquelle le pourcentage d'avancement d'une graine est calcule
    seuilPourcentageMouvementDeGraine = 5 # Pourcentage d'avancement maximum d'une graine pour que celle-ci soit consideree comme stable.

    distancesAnciensNouveauxBarycentres = numpy.zeros(K)
    pourcentageAvancementGraine = numpy.zeros(K)


    stabiliteBoucle = False

    # Debut boucle
    while(not stabiliteBoucle):

        # On stocke les anciennes positions des graines afin de pouvoir determiner plus tard la stabilite du mouvement des graines (en calculant le taux d'avancement des graines)
        oldBarycentres_bleu = cluster_bleu.copy()
        oldBarycentres_rouge = cluster_rouge.copy()
        oldBarycentres_vert = cluster_vert.copy()

        # Calcul du barycentre des cluster et attribution des nouvelles positions des graines
        SetBarycentreAuxGraines(groupe,bleu, vert, rouge, cluster_bleu, cluster_rouge,cluster_vert, K, nb_pixels)

        # Calcul de stabilite du mouvement des graines par rapport a leur ancienne position
        estStable = True
        for graine in range(0, K):
            distancesAnciensNouveauxBarycentres[graine] = CalculeModule(oldBarycentres_vert[graine], oldBarycentres_rouge[graine], oldBarycentres_bleu[graine], cluster_vert[graine],cluster_rouge[graine], cluster_bleu[graine])
            pourcentageAvancementGraine[graine] = (distancesAnciensNouveauxBarycentres[graine] * 100) / distanceReferenceDiagonaleCoordonneesCouleurs
            if(pourcentageAvancementGraine[graine] > seuilPourcentageMouvementDeGraine):
                estStable = False
        stabiliteBoucle = estStable


        # Attribution des pixels aux clusters dont les graines sont les plus proches des pixels
        AttributionKMeans(groupe,bleu, rouge, vert, cluster_bleu, cluster_rouge, cluster_vert, K, nb_pixels)

        nbPixelsParCluster = numpy.zeros(K)
        for pixel in range(0,nb_pixels):
            te = int(groupe[pixel, 0])
            nbPixelsParCluster[te] += 1


    exit(0)


    # Calculer le barycentre des points de chaque cluster et l'attribuer aux graines de chaque cluster
    #      ---> Utiliser --> function SetBarycentreAuxGraines(groupe, bleu, vert, rouge, cluster_bleu, cluster_rouge, cluster_vert): Liste  => La fonction doit calculer la nouvelle position de chaque graine (cluster_bleu, cluster_rouge, cluster_vert) en fonction du barycentre des points de chaque cluster.

    # Effectuer une attribution K-Means (Chaque pixel est redirige vers le cluster dont la graine est la plus proche)



    # Boucle Do While (tant que le nombre d'elements dans chaque cluster n'a pas augmente de plus de P % (P = 5 % ?))


        # Calculer le barycentre des points de chaque cluster et l'attribuer aux graines de chaque cluster
        #      ---> Utiliser --> function SetBarycentreAuxGraines(groupe, bleu, vert, rouge, cluster_bleu, cluster_rouge, cluster_vert): Liste  => La fonction doit calculer la nouvelle position de chaque graine (cluster_bleu, cluster_rouge, cluster_vert) en fonction du barycentre des points de chaque cluster.

        # Effectuer une attribution K-Means (Chaque pixel est redirige vers le cluster dont la graine est la plus proche)




        #######################
    ####### Fin code eleves #######
        #######################




    # Instructions du prof ----

    #La, c'est a vous d'ecrire le code de la boucle principale
    #Votre code doit faire evoluer les tableaux groupe, cluster_bleu, cluster_rouge et cluster_vert
    #...

    #Fin de l'algo, on affiche les resultats

# -------------------------





    #On change le format de groupe afin de le rammener au format de l'image d'origine
    groupe=numpy.reshape(groupe, (imagecolor.shape[0], imagecolor.shape[1]))

    #On change chaque pixel de l'image selon le cluster auquel il appartient
    #Il prendre comme nouvelle valeur la position moyenne du cluster
    for i in range(0, imagecolor.shape[0]):
        for j in range(0, imagecolor.shape[1]):
            imagecolor[i,j,0] = (cluster_bleu[groupe[i,j]])
            imagecolor[i,j,1] = (cluster_vert[groupe[i,j]])
            imagecolor[i,j,2] = (cluster_rouge[groupe[i,j]])



	
    # On affiche une fenetre avec l'image
    cv2.namedWindow("sortie")
    #On sort quand l'utilisateur appuie sur une touche
    cv2.imshow("sortie", imagecolor)
    key = cv2.waitKey(0)



if __name__ == "__main__":
    main()
