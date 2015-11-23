""" Question 9 """
import networkx as nx
import matplotlib.pyplot as plt

def run():
    g = {
        'f': {r'$\theta$', 'X'},
        'Y': 'f'
    }
    gp = nx.DiGraph(g)
    pos = nx.spring_layout(gp)
    nx.draw_networkx_nodes(gp, pos, node_color='w', node_size=1000)
    nx.draw_networkx_edges(gp, pos)
    nx.draw_networkx_labels(gp, pos, font_size=16)
    plt.axis('off')
