import os
import xml.etree.ElementTree as ET
import cv2

DOSSIER_IMAGES = r"C:\Users\moham\Desktop\bip\haha\RGB"
DOSSIER_XML = r"C:\Users\moham\Desktop\bip\haha\annotations"

DOSSIER_SORTIE = r"C:\Users\moham\Desktop\bip\haha\Dataset_Pret"
DOSSIER_SORTIE_IMAGES = os.path.join(DOSSIER_SORTIE, "images")
DOSSIER_SORTIE_XML = os.path.join(DOSSIER_SORTIE, "annotations")

os.makedirs(DOSSIER_SORTIE_IMAGES, exist_ok=True)
os.makedirs(DOSSIER_SORTIE_XML, exist_ok=True)

NOUVELLE_TAILLE = 800 

print(" Lancement du prétraitement complet...")
print(f" Les nouvelles données seront sauvegardées ici : {DOSSIER_SORTIE}\n")

fichiers_xml = [f for f in os.listdir(DOSSIER_XML) if f.endswith('.xml')]
compteur_succes = 0


for fichier_xml in fichiers_xml:
    nom_image = fichier_xml.replace('.xml', '.tif')
    chemin_image_entree = os.path.join(DOSSIER_IMAGES, nom_image)
    chemin_xml_entree = os.path.join(DOSSIER_XML, fichier_xml)
    
    
    if os.path.exists(chemin_image_entree):
        
        image = cv2.imread(chemin_image_entree)
        arbre_xml = ET.parse(chemin_xml_entree)
        racine = arbre_xml.getroot()
        
       
        ancienne_hauteur, ancienne_largeur = image.shape[:2]
        ratio_x = NOUVELLE_TAILLE / ancienne_largeur
        ratio_y = NOUVELLE_TAILLE / ancienne_hauteur
        
        
        image_redimensionnee = cv2.resize(image, (NOUVELLE_TAILLE, NOUVELLE_TAILLE))
       
        lab = cv2.cvtColor(image_redimensionnee, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l_clahe = clahe.apply(l)
        image_amelioree = cv2.merge((l_clahe, a, b))
        image_amelioree = cv2.cvtColor(image_amelioree, cv2.COLOR_LAB2BGR)
        
        image_finale = cv2.GaussianBlur(image_amelioree, (5, 5), 0)
        
        for obj in racine.findall('object'):
            bndbox = obj.find('bndbox')
            
            ancien_xmin = float(bndbox.find('xmin').text)
            ancien_ymin = float(bndbox.find('ymin').text)
            ancien_xmax = float(bndbox.find('xmax').text)
            ancien_ymax = float(bndbox.find('ymax').text)
            
            nouveau_xmin = max(0, int(ancien_xmin * ratio_x))
            nouveau_ymin = max(0, int(ancien_ymin * ratio_y))
            nouveau_xmax = min(NOUVELLE_TAILLE, int(ancien_xmax * ratio_x))
            nouveau_ymax = min(NOUVELLE_TAILLE, int(ancien_ymax * ratio_y))
            
            bndbox.find('xmin').text = str(nouveau_xmin)
            bndbox.find('ymin').text = str(nouveau_ymin)
            bndbox.find('xmax').text = str(nouveau_xmax)
            bndbox.find('ymax').text = str(nouveau_ymax)

        size = racine.find('size')
        if size is not None:
            if size.find('width') is not None: size.find('width').text = str(NOUVELLE_TAILLE)
            if size.find('height') is not None: size.find('height').text = str(NOUVELLE_TAILLE)

        chemin_image_sortie = os.path.join(DOSSIER_SORTIE_IMAGES, nom_image)
        chemin_xml_sortie = os.path.join(DOSSIER_SORTIE_XML, fichier_xml)
        
        cv2.imwrite(chemin_image_sortie, image_finale)
        arbre_xml.write(chemin_xml_sortie)
        
        compteur_succes += 1
        if compteur_succes % 10 == 0:
            print(f" {compteur_succes} images traitées...")

print(f"\n TERMINÉ ! {compteur_succes} images et annotations ont été préparées avec succès.")
print(f"Va vérifier ton dossier : {DOSSIER_SORTIE}")