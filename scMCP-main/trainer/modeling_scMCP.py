from models.scMCP import scMCP
from trainer.co_express_networks import get_similarity_network,GeneSimNetwork              

def create_scMCP(pretrained=False, **kwargs):
    config =  {
            'comb_num' : 1,
            'split_key' :'celltype_split1',
            'top_ranked_genes' : 2048,
            'hidden_layer_sizes' : [128],
            'z_dimension' : 128,
            'adaptor_layer_sizes' : [128],
            'comb_dimension' : 64, 
            'drug_dimension': 768,
            'dr_rate' : 0.05,
            'n_epochs' : 100,
            'lr' :3e-4, 
            'weight_decay' : 1e-8,
            'scheduler_factor' : 0.7,
            'scheduler_patience' : 1,
            'n_genes' : 20,
            'loss' : ['ZINLL'], 
            'obs_key' : 'cov_drug',
            'pos_emb_graph' :'go',#go,co_expression,None
            'dataset' :'tahoe',
            'gene_emb_adapt':False,
            'da_mode':True,
            'pretrain_vae':False
        }    
    coexpress_threshold = 0.1
    num_similar_genes_co_express_graph = 50
    
    
    with open('/home/MBDAI206AA201/jupyter/yhz/sc/MOMDGDP-main/dataset/gene_list.txt', 'r') as f:
        gene_list = [line.strip() for line in f if line.strip()]
    data_path = '/home/MBDAI206AA201/jupyter/yhz/sc/MOMDGDP-main/embeddings/'
    dataset_name = config['dataset']
    split = config['split_key']
    train_gene_set_size = len(gene_list)
    node_map = {x: it for it, x in enumerate(gene_list)}
    edge_list = get_similarity_network(network_type = config['pos_emb_graph'], adata = None, 
                                                       threshold = coexpress_threshold, 
                                                       k = num_similar_genes_co_express_graph, gene_list =gene_list, data_path =data_path, data_name = dataset_name, split =split, seed =2025, train_gene_set_size = train_gene_set_size)
    sim_network = GeneSimNetwork(edge_list, gene_list, node_map = node_map)
    model = scMCP(
                num_genes=config['top_ranked_genes'],
                     uncertainty=False,
                     num_gnn_layers=None,
                     decoder_hidden_size=None,
                     num_gene_gnn_layers=None,
                     input_genes_ens_ids=None,
            gene_mask = None,
                     scfm_genes_ens_ids=None,
            da_mode = False,
            gene_emb_adapt=False,
                coexpress_network = sim_network,
                drug_dim=config['drug_dimension'],
                     hidden_size = config['z_dimension'],
                pos_emb_graph = config['pos_emb_graph'],
                     grn_node2vec_file='/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/downstream_tasks/PRCEdrug/GraphEmbedding/node2vec/emb_grn/grn_emb_total.pkl',
                     ppi_node2vec_file='/home/MBDAI206AA201/jupyter/yhz/sc/scdata/GeneCompass-main/downstream_tasks/PRCEdrug/GraphEmbedding/node2vec/emb_ppi/ppi_emb_total.pkl',
                     model_type = 'ppi_grn_mode')
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
