(ns ribelo.malmo
  (:require
   [ribelo.haag :as h]
   [criterium.core :refer [quick-bench]]
   [ribelo.visby.math :as math]))

(defn train-test-split
  ([^double p coll]
   (train-test-split p coll {:shuffle? false}))
  ([^double p coll {:keys [shuffle?]}]
   (let [c (count coll)
         n (math/floor (* p c))]
     (split-at n (if shuffle? (shuffle coll) coll)))))

(defn maps->2d-vec [ks maps]
  (->> maps
       (into []
         (map (fn [m]
                (reduce (fn [acc k] (conj acc (get m k))) [] ks))))))

(defn maps->1d-vec [ks maps]
  (->> maps
       (into []
             (comp
              (map (fn [m]
                     (reduce (fn [acc k] (conj acc (get m k))) [] ks)))
              cat))))

(defn maps->2d-arr
  ([ks maps]
   (maps->2d-arr :double ks maps))
  ([atype ks maps]
   (->> maps
        (into []
              (map (fn [m]
                     (into-array (h/dtype atype) (reduce (fn [acc k] (conj acc (get m k))) [] ks)))))
        (into-array))))

(defn maps->1d-arr
  ([ks maps]
   (maps->1d-arr :double ks maps))
  ([atype ks maps]
   (into-array (h/dtype atype) (maps->1d-vec ks maps))))

(defmulti coll->x ;;TODO
  {:arglists '([coll] [ks coll] [dtype coll] [dtype ks coll] [flat? dtype ks coll])}
  (fn
    ([coll]                [(type coll) (when (sequential? coll) (type (first coll)))])
    ([_ coll]              [(type coll) (when (sequential? coll) (type (first coll)))])
    ([dtype ks coll]       [(type coll) (when (sequential? coll) (type (first coll)))])
    ([flat? dtype ks coll] [(type coll) (when (sequential? coll) (type (first coll)))])))

(defmethod coll->x [java.util.Collection java.util.Map]
  ([coll]
   (coll->x (keys (first coll)) coll))
  ([ks coll]
   (maps->2d-vec ks coll))
  ([atype ks coll]
   (maps->2d-arr atype ks coll))
  ([flat? atype ks coll]
   (maps->1d-arr atype ks coll)))

(defmethod coll->x [java.util.Collection java.util.Collection]
  ([coll] coll)
  ([atype coll]
   (into-array (into [] (map #(into-array (h/dtype atype) %)) coll)))
  ([flat? atype coll]
   (into-array (h/dtype atype) (into [] cat coll))))

(defmethod coll->x [h/double-double-array-type nil]
  ([coll] coll)
  ([atype coll]
   (if-not (= atype h/double-type)
     (into-array (h/dtype atype) coll)
     coll))
  ([flat? atype coll]
   (into-array (h/dtype atype) (into [] cat coll))))

(defmethod coll->x [h/float-float-array-type nil]
  ([coll] coll)
  ([atype coll]
   (if-not (= atype h/float-type)
     (into-array (h/dtype atype) coll)
     coll))
  ([flat? atype coll]
   (into-array (h/dtype atype) (into [] cat coll))))

(defmulti coll->y
  (fn
    ([coll]         [(type coll) (type (first coll))])
    ([_ coll]       [(type coll) (type (first coll))])
    ([dtype k coll] [(type coll) (type (first coll))])))

(defmethod coll->y [java.util.Collection java.util.Map]
  ([coll]
   (coll->y (keys (first coll)) coll))
  ([k coll]
   (into [] (map (fn [m] (get k m))) coll))
  ([atype k coll]
   (->> coll
        (into [] (map (fn [m] (get k m))))
        (into-array (h/dtype atype)))))

(defmethod coll->y [java.util.Collection java.lang.Number]
  ([coll] coll)
  ([atype coll]
   (into-array (h/dtype atype) coll)))

(defmethod coll->y [h/double-array-type nil]
  ([coll] coll)
  ([atype coll]
   (into-array (h/dtype atype) coll)))

(defmethod coll->y [h/float-array-type nil]
  ([coll] coll)
  ([atype coll]
   (into-array (h/dtype atype) coll)))
