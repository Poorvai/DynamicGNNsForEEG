{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "3adad4e0-3767-4151-b813-bd4fd0e6403c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import scipy.sparse as sp\n",
    "import torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "50db720a-840f-4af1-b9c1-01ae85890f28",
   "metadata": {},
   "outputs": [],
   "source": [
    "def encode_onehot(labels):\n",
    "    classes = set(labels)\n",
    "    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in\n",
    "                    enumerate(classes)}\n",
    "    labels_onehot = np.array(list(map(classes_dict.get, labels)),\n",
    "                             dtype=np.int32)\n",
    "    return labels_onehot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7b77e015-eb91-40e8-a338-566f35a7d5fd",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
