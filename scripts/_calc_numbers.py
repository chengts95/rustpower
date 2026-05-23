ASM = {'39':{'V0':29.7,'V1':6.19,'V2':1.84},'118':{'V0':102.0,'V1':16.7,'V2':6.03},'9241':{'V0':13745,'V1':4635,'V2':641}}
PAPER={'39':{'V0':29.7,'V1':6.19,'V2':1.84},'118':{'V0':102.0,'V1':16.7,'V2':6.03},'9241':{'V0':9155,'V1':2566,'V2':627}}
RF={'39':3.64,'118':8.73,'9241':2889}; BS={'39':0.76,'118':1.82,'9241':353}
ANA=7857; FAC=8731; SYM=1033; N=5; OLD_KLU=3305+406

print('=== 1. PER-ITERATION FRACTIONS (latest benchmark) ===')
for s in ['39','118','9241']:
    klu=RF[s]+BS[s]
    for v in ['V0','V1','V2']:
        a=ASM[s][v]; print(f'  {s:6} {v}: asm={a:.1f}us  KLU={klu:.2f}us  frac={a/(a+klu)*100:.1f}%')
    print()

print('=== PEGASE: paper text vs old data vs new data ===')
for v,pv in [('V0',71),('V1',41),('V2',14)]:
    po=PAPER['9241'][v]; pn=ASM['9241'][v]; kn=RF['9241']+BS['9241']
    print(f'  {v}: paper_text={pv}%  old_data_calc={po/(po+OLD_KLU)*100:.1f}%  new_data_calc={pn/(pn+kn)*100:.1f}%')

print()
print('=== 2. PER-SOLVE FRACTIONS PEGASE N=5 ===')
ki=ANA+FAC; ks=(N-1)*RF['9241']+N*BS['9241']
print(f'  KLU_init={ki/1e3:.2f}ms  KLU_steady={ks/1e3:.2f}ms  KLU_total={ki/1e3+ks/1e3:.2f}ms')
for v,sym in [('V0',0),('V1',0),('V2',SYM)]:
    a=N*ASM['9241'][v]+sym; tot=a+ki+ks
    print(f'  {v}: asm_total={a/1e3:.2f}ms  grand_total={tot/1e3:.2f}ms  frac={a/tot*100:.1f}%')

print()
print('=== 3. SCALING: V0 PEGASE assembly% vs N_iter ===')
a0=ASM['9241']['V0']
for Ni in [3,5,7,10,15,20,50]:
    kt=ki+(Ni-1)*RF['9241']+Ni*BS['9241']; at=Ni*a0
    print(f'  N={Ni:3}: {at/(at+kt)*100:.1f}%')
print(f'  N=inf: {a0/(a0+RF["9241"]+BS["9241"])*100:.1f}%  (per-iter asymptote)')

print()
print('=== 4. SYMBOLIC BUILD vs FILL (V2 PEGASE) ===')
fill=ASM['9241']['V2']
print(f'  symbolic={SYM}us  fill={fill}us  ratio={SYM/fill:.2f}x')

print()
print('=== 5. TAB:ASMONLY: current paper vs benchmark ===')
for s in ['39','118','9241']:
    for v in ['V0','V1','V2']:
        p=PAPER[s][v]; n=ASM[s][v]
        flag=' <-- STALE' if abs(p-n)/max(p,n)>0.05 else ''
        print(f'  {s:6} {v}: paper={p:.1f}  benchmark={n:.1f}{flag}')
    v0p=PAPER[s]['V0']; v2p=PAPER[s]['V2']; v1p=PAPER[s]['V1']
    v0n=ASM[s]['V0'];   v2n=ASM[s]['V2'];   v1n=ASM[s]['V1']
    print(f'         V0/V2: paper={v0p/v2p:.1f}x  benchmark={v0n/v2n:.1f}x  |  V1/V2: paper={v1p/v2p:.1f}x  benchmark={v1n/v2n:.1f}x')
    print()

print()
print('=== 6. SPACE ESTIMATE PEGASE 9241 ===')
# nnz(Ybus) estimated by timing ratio vs IEEE 118 (nnz_Ybus=688, well-known)
nnz_y=int(688*(641/6.03)); nnz_j=nnz_y*2; ns=18280
print(f'  nnz(Ybus) ~ {nnz_y:,}  (from IEEE-118 nnz=688 scaled by fill-time ratio 641/6.03)')
print(f'  nnz(J)    ~ {nnz_j:,}  (estimated 2x nnz_Ybus for 4-block structure)')
print(f'  n_state   ~ {ns:,}')
sens=2*nnz_y*16         # 2 complex sensitivity matrices (values only)
sens_idx=2*(nnz_y*4+(9241+1)*4)  # CSC indices
vm=nnz_j*8             # LS2G value_map_ pointers
jp=(ns+1)*8+nnz_j*8+nnz_y*16  # V2 JacobianPattern
jv=nnz_j*8            # V2 j_values
print(f'\n  V0 per-iter TRANSIENT allocations:')
print(f'    2x sensitivity matrices (values): {sens//1024:,} KB = {sens/1024/1024:.1f} MB')
print(f'    index arrays for above:           {sens_idx//1024:,} KB')
print(f'    Total transient per iter:         {(sens+sens_idx)//1024:,} KB ~ {(sens+sens_idx)/1024/1024:.1f} MB  (freed after each iter)')
print(f'\n  LS2G PERSISTENT extra vs V2:')
print(f'    value_map_ (O(nnz_J) pointers):  {vm//1024:,} KB = {vm/1024/1024:.2f} MB')
print(f'\n  V2 PERSISTENT (one-time):')
print(f'    JacobianPattern total:            {jp//1024:,} KB = {jp/1024/1024:.2f} MB')
print(f'    j_values (numeric buffer):        {jv//1024:,} KB')
print(f'    Per-iter new alloc:               0 KB')
print(f'\n  SAVINGS:')
print(f'    vs V0 per iter: eliminate {(sens+sens_idx)//1024:,} KB transient = {(sens+sens_idx)/(sens+sens_idx+jp+jv)*100:.0f}% of V0 total working set')
print(f'    vs LS2G: eliminate value_map_ {vm//1024:,} KB + per-iter sens {sens//1024:,} KB/iter')
