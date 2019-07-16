(ns ribelo.malmo
  (:require
   [ribelo.visby.math :as math]))

(defn train-test-split
  ([^double p coll]
   (train-test-split p coll {:shuffle? false}))
  ([^double p coll {:keys [shuffle?]}]
   (let [c (count coll)
         n (math/floor (* p c))]
     (split-at n (if shuffle? (shuffle coll) coll)))))

(defn data->x
  ([data]
   (data->x (keys (first data)) Double/TYPE data))
  ([ks data]
   (data->x (keys (first data)) data))
  ([ks atype data]
   (into-array
    (mapv (fn [m]
            (into-array atype (reduce (fn [acc k] (conj acc (get m k))) [] ks))) data))))

(defn data->y
  ([k data]
   (data->y k Double/TYPE data))
  ([k atype data]
   (into-array atype (mapv k data))))