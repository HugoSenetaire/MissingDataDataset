
from .composite_generators import dic_composite_generators
from .simple_generators import dic_simple_generators
from .stats_generators import dic_stats_generator
from .abstract_mask_generator import NoneMaskGenerator

liste_dic = [dic_simple_generators, dic_stats_generator, dic_composite_generators]
#Check that the names of the dictionaries are different
for dic_1, dic_2 in [(dic_1, dic_2) for dic_1 in liste_dic for dic_2 in liste_dic]:
    if dic_1 is not dic_2:
        for key in dic_1:
            assert key not in dic_2, "Two dictionaries have the same name"

dic_mask_generators = {**dic_simple_generators, **dic_stats_generator, **dic_composite_generators}