import csv

with open('notebooks/runs/detect/train_ball_y12s/results.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print('\n' + '='*50)
print('ANÁLISE DO TREINO - BALL DETECTION (YOLOv12s)')
print('='*50)

last = rows[-1]
print('\n=== MÉTRICAS FINAIS (Epoch 50) ===')
precision = float(last['metrics/precision(B)'])
recall = float(last['metrics/recall(B)'])
map50 = float(last['metrics/mAP50(B)'])
map5095 = float(last['metrics/mAP50-95(B)'])

print(f'Precision:    {precision:.3f} ({precision*100:.1f}%)')
print(f'Recall:       {recall:.3f} ({recall*100:.1f}%)')
print(f'mAP50:        {map50:.3f} ({map50*100:.1f}%)')
print(f'mAP50-95:     {map5095:.3f} ({map5095*100:.1f}%)')
print(f'\nVal Box Loss: {float(last["val/box_loss"]):.3f}')
print(f'Val Cls Loss: {float(last["val/cls_loss"]):.3f}')

print('\n=== EVOLUÇÃO DO TREINO ===')
print(f'Epoch 1  mAP50: {float(rows[0]["metrics/mAP50(B)"]):.3f}')
print(f'Epoch 10 mAP50: {float(rows[9]["metrics/mAP50(B)"]):.3f}')
print(f'Epoch 25 mAP50: {float(rows[24]["metrics/mAP50(B)"]):.3f}')
print(f'Epoch 40 mAP50: {float(rows[39]["metrics/mAP50(B)"]):.3f}')
print(f'Epoch 50 mAP50: {float(rows[-1]["metrics/mAP50(B)"]):.3f}')

print('\n=== DIAGNÓSTICO ===')

if recall < 0.4:
    print(f'⚠️  RECALL BAIXO ({recall*100:.1f}%):')
    print('   - O modelo está PERDENDO muitas bolas!')
    print(f'   - Não detecta {(1-recall)*100:.0f}% das bolas no dataset de validação')
    print('   - Soluções:')
    print('     1. Treinar com mais epochs (100-150 em vez de 50)')
    print('     2. Reduzir confidence threshold (de 0.25 para 0.15 ou 0.1)')
    print('     3. Usar imgsz maior (1280 em vez de 640)')
    print('     4. Verificar qualidade do dataset (bolas pequenas/oclusas)')
else:
    print(f'✅ RECALL OK ({recall*100:.1f}%)')

if precision > 0.7:
    print(f'\n✅ PRECISION BOA ({precision*100:.1f}%):')
    print('   - Quando detecta, geralmente está correto')
    print('   - Poucos falsos positivos')

if map50 < 0.5:
    print(f'\n⚠️  mAP50 RAZOÁVEL ({map50*100:.1f}%):')
    print('   - Performance geral OK mas pode melhorar muito')
    print('   - Objetivo ideal: 50-70% mAP50 para bola')
    print('   - Com mais treino pode chegar a 50-60%')

# Find best epoch
best_map50 = max(float(r['metrics/mAP50(B)']) for r in rows)
best_epoch = [i+1 for i, r in enumerate(rows) if float(r['metrics/mAP50(B)']) == best_map50][0]

print('\n=== MELHOR ÉPOCA ===')
print(f'Epoch {best_epoch}: mAP50 = {best_map50:.3f}')
if best_epoch < 50:
    print(f'⚠️  Melhor resultado foi no epoch {best_epoch}, não no 50!')
    print(f'   Possível overfitting após epoch {best_epoch}')
    print(f'   Deveria usar weights do epoch {best_epoch}')

print('\n=== RECOMENDAÇÕES ===')
print('1. URGENTE: Reduzir confidence threshold para 0.1-0.15 no main.py')
print('   - Linha: ball_result = ball_detection_model(frame, imgsz=640, conf=0.1, verbose=False)[0]')
print('2. Retreinar com 100-150 epochs para melhorar recall')
print('3. Considerar usar imgsz=1280 para detectar bolas menores')
print('4. Verificar se há early stopping - pode ter parado antes do ótimo')

print('\n' + '='*50)

