import pandapower.networks as pn
net = pn.case118()
print("gen index:", list(net.gen.index[:10]), "... total:", len(net.gen))
print("poly_cost element values:", sorted(net.poly_cost['element'].tolist()))
print("poly_cost et values:", net.poly_cost['et'].unique().tolist())
# Check: are all gen indices present in poly_cost?
gen_idx_set = set(net.gen.index)
pc_gen_set = set(net.poly_cost[net.poly_cost['et']=='gen']['element'])
print("gen indices NOT in poly_cost:", gen_idx_set - pc_gen_set)
print("poly_cost elements NOT in gen:", pc_gen_set - gen_idx_set)
