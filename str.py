import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# la generation des signaux #
def awgn(signal, snr_db):
    snr_linear = 10**(snr_db / 10)
    power_signal = np.mean(np.abs(signal)**2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape))
    return signal + noise

def generate_bpsk(n):
    return 2 * np.random.randint(0, 2, n) - 1

def generate_qpsk(n):
    bits = np.random.randint(0, 2, (n, 2))
    return (2*bits[:, 0] - 1 + 1j*(2*bits[:, 1] - 1)) / np.sqrt(2)

def generate_8psk(n):
    bits = np.random.randint(0, 8, n)
    return np.exp(1j * 2 * np.pi * bits / 8)

def generate_qam(M, n):
    m = int(np.sqrt(M))
    re = 2 * (np.random.randint(0, m, n) - (m - 1) / 2)
    im = 2 * (np.random.randint(0, m, n) - (m - 1) / 2)
    return (re + 1j * im) / np.sqrt((2 / 3) * (M - 1))

# interface streamlit #
st.title("Détection automatique de modulation numérique")

modulation = st.selectbox("Choisissez une modulation", ['BPSK', 'QPSK', '8-PSK', '4-QAM', '16-QAM', '64-QAM'])
snr = st.slider("SNR (dB)", 0, 30, 10)
n_symbols = st.slider("Nombre de symboles", 100, 3000, 1000)
kernel_type = st.selectbox("Choisissez le noyau SVM", ['linear', 'poly', 'rbf'])

# ------------------ GÉNÉRATION DU SIGNAL ------------------ #
if modulation == 'BPSK':
    signal = generate_bpsk(n_symbols)
elif modulation == 'QPSK':
    signal = generate_qpsk(n_symbols)
elif modulation == '8-PSK':
    signal = generate_8psk(n_symbols)
elif modulation == '16-QAM':
    signal = generate_qam(16, n_symbols)
else:
    signal = generate_qam(64, n_symbols)

signal_noisy = awgn(signal, snr)
X_input = np.column_stack((signal_noisy.real, signal_noisy.imag))

#plot de la constellation#
st.subheader("Constellation du signal bruité")
fig1, ax1 = plt.subplots()
ax1.scatter(X_input[:, 0], X_input[:, 1], s=5, alpha=0.5)
ax1.axhline(0, color='black')
ax1.axvline(0, color='black')
ax1.set_title(f"Constellation - {modulation}")
ax1.set_xlabel("In-phase (I)")
ax1.set_ylabel("Quadrature (Q)")
ax1.grid(True)
st.pyplot(fig1)

#la phase d'entrainement  #
def generate_dataset_per_mod():
    mod_funcs = {
        'BPSK': generate_bpsk,
        'QPSK': generate_qpsk,
        '8-PSK': generate_8psk,
        '16-QAM': lambda n: generate_qam(16, n),
        '64-QAM': lambda n: generate_qam(64, n)
    }
    X, y = [], []
    for name, func in mod_funcs.items():
        s = func(n_symbols)
        s_n = awgn(s, snr)
        X.append(np.column_stack((s_n.real, s_n.imag)))
        y += [name] * n_symbols
    return np.vstack(X), np.array(y)

X_train, y_train = generate_dataset_per_mod()

# affichage des donnees d'entrainement #
st.subheader("Aperçu des données d'entraînement générées")
st.write("Extrait de X_train (composantes I et Q):")
st.write(X_train[:10])
st.write("Extrait de y_train (modulations associées):")
st.write(y_train[:10])

if st.checkbox("Afficher toutes les données d'entraînement"):
    st.write(X_train)
    st.write(y_train)

# la classification #
degree = 3
clf = svm.SVC(kernel=kernel_type, degree=degree, gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_input)

# Classifieur secondaire pour frontières
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
clf_encoded = svm.SVC(kernel=kernel_type, degree=degree, gamma='scale')
clf_encoded.fit(X_train, y_train_encoded)

# les predictions #
st.subheader("Prédiction de la modulation")
unique, counts = np.unique(y_pred, return_counts=True)
probas = dict(zip(unique, counts / len(y_pred)))

for mod, prob in probas.items():
    st.write(f"**{mod}** : {prob*100:.2f}%")

st.success(f"Modulation prédite : **{max(probas, key=probas.get)}**")

#les plots des predictions#
st.subheader("Visualisation des prédictions par couleur")
fig2, ax2 = plt.subplots()
colors = {'BPSK':'blue', 'QPSK':'red', '8-PSK':'green', '4-QAM':'purple', '16-QAM':'orange', '64-QAM':'cyan'}

for label in np.unique(y_pred):
    idx = np.where(y_pred == label)
    ax2.scatter(X_input[idx, 0], X_input[idx, 1], s=5, alpha=0.5, label=label, color=colors[label])

ax2.axhline(0, color='black')
ax2.axvline(0, color='black')
ax2.set_title("Classification par SVM")
ax2.set_xlabel("In-phase (I)")
ax2.set_ylabel("Quadrature (Q)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

#  Affichage des frontieres de decision entre les symboles des constillations#
st.subheader("Frontières de décision du SVM")
h = 0.05
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf_encoded.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig3, ax3 = plt.subplots()
ax3.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.rainbow)

for label in np.unique(y_train):
    idx = np.where(y_train == label)[0]
    ax3.scatter(X_train[idx, 0], X_train[idx, 1], label=label, s=10, alpha=0.6, color=colors[label])

ax3.set_xlabel("In-phase (I)")
ax3.set_ylabel("Quadrature (Q)")
ax3.set_title(f"Frontières de décision - Noyau {kernel_type}")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)

# La comparaison entre les noyaux SVM #
def comparer_noyaux(X_train, y_train, X_test, y_test):
    kernels = ['linear', 'poly', 'rbf']
    scores = {}
    for k in kernels:
        clf = svm.SVC(kernel=k, degree=3, gamma='scale')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        scores[k] = acc
    return scores

st.subheader("Comparaison des noyaux SVM")
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.3, random_state=42)
scores = comparer_noyaux(X_train_split, y_train_split, X_test_split, y_test_split)

for k, acc in scores.items():
    st.write(f"Noyau **{k}** : {acc*100:.2f}% de précision")

fig4, ax4 = plt.subplots()
ax4.bar(scores.keys(), scores.values(), color=['blue', 'orange', 'green'])
ax4.set_ylabel("Précision")
ax4.set_ylim(0, 1)
ax4.set_title("Performance des noyaux SVM")
st.pyplot(fig4)

# Cette partie pour afficher la matrice de confusion pour chaque noyau  #
st.subheader("Matrices de confusion pour chaque noyau SVM")

def plot_confusions_by_kernel(X_train, y_train):
    kernels = ['linear', 'poly', 'rbf']
    X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    for k in kernels:
        clf = svm.SVC(kernel=k, degree=3, gamma='scale')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        cm = confusion_matrix(y_te, y_pred, labels=np.unique(y_train))
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_train))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        ax.set_title(f"Matrice de confusion - Noyau {k}")
        st.pyplot(fig)

plot_confusions_by_kernel(X_train, y_train)
