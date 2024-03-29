(ns ribelo.malmo.xgboost
  (:require
   [ribelo.haag :as h]
   [ribelo.malmo :as ml])
  (:import
   (ml.dmlc.xgboost4j.java XGBoost
                           Booster
                           DMatrix)))

(defn ->array [coll]
  (float-array (into [] cat coll)))

(defprotocol Dmatrix
  (dmatrix [x] [x y]))

(extend-protocol Dmatrix
  clojure.lang.PersistentArrayMap
  (dmatrix [{:keys [x y]}]
    (if y
      (doto
       (DMatrix.
        (->array x)
        (count y)
        (count (first x)))
        (.setLabel (float-array y)))
      (DMatrix.
       (->array x)
       (count x)
       (count (first x)))))
  clojure.lang.PersistentVector
  (dmatrix
    ([x]
     (DMatrix.
      (->array x)
      (count x)
      (count (first x))))
    ([x y]
     (doto
      (DMatrix.
       (->array x)
       (count y)
       (count (first x)))
       (.setLabel (float-array y)))))
  java.lang.String
  (dmatrix [path]
    (DMatrix. path)))

(extend-type (Class/forName "[[D")
  Dmatrix
  (dmatrix
    ([x]
     (DMatrix.
      (->array x)
      (alength ^doubles x)
      (alength ^doubles (aget ^doubles x 0))))
    ([x y]
     (doto
      (DMatrix.
       (->array x)
       (alength ^doubles x)
       (alength ^doubles (aget ^doubles x 0)))
       (.setLabel (->array y))))))

(defn- keywords->str
  [m]
  (reduce-kv
   (fn [acc k v]
     (assoc acc (clojure.string/replace (name k) "-" "_") v))
   {}
   m))

(defn fit [^DMatrix dmatrix {:keys [params rounds watches early-stopping booster]}]
  (let [opts    (keywords->str params)
        watch   (keywords->str watches)
        metrics (make-array Float/TYPE (count watch) rounds)]
    (XGBoost/train dmatrix opts
                   rounds watch
                   metrics
                   nil nil
                   early-stopping
                   booster)))

(def train fit)

(defmethod ml/predict [Booster DMatrix]
  ([model coll]
   (into [] cat (.predict model coll))))

(defmethod ml/predict [Booster h/double-double-array-type]
  ([model arr]
   (into [] cat (.predict model (dmatrix (->array arr))))))

(defmethod ml/predict [Booster clojure.lang.PersistentVector]
  ([model arr]
   (into [] cat (.predict model (dmatrix (->array arr))))))

(defmethod ml/save-model Booster
  [model ^String path] (.saveModel model path))
