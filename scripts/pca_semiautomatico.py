import cv2 as cv
import numpy as np
import pca
import argparse

def _get_Args():
     parser = argparse.ArgumentParser()
     parser.add_argument("file", help="File to reduce")
     parser.add_argument("features",type=int, help="Number of features to reduce")
     parser.add_argument("-o", "--outdir", help="Output directory", nargs='?', const='', default='')
     return parser.parse_args()


def _main(args):
     arquivo_a_reduzir = args.file
     numero_de_features_apos_PCA = args.features
     cnn_flow = pca.read_cnnf_file(arquivo_a_reduzir)
     cnn_flow_padronizado = pca.cnn_flow_para_PCA(cnn_flow)
     numero_de_samples = len(cnn_flow_padronizado)
     cnn_flow_conformado = pca.conformar_cnn_flow_para_PCA(cnn_flow_padronizado)
     auto_vetores, media = pca.baseground_PCA(cnn_flow_conformado,numero_de_features_apos_PCA)
     pca.write_pca_baseground("autovetores", arquivo_a_reduzir, auto_vetores, args.outdir)
     pca.write_pca_baseground("vetor_media", arquivo_a_reduzir, media, args.outdir)
     projecao = pca.projecao_PCA(cnn_flow_padronizado, media, auto_vetores)
     pca.write_pca_reduction(arquivo_a_reduzir,projecao, args.outdir)
	 
if __name__ == '__main__':
     args = _get_Args()
     _main(args)
