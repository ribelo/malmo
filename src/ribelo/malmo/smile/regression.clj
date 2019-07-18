(ns ribelo.malmo.smile.regression
  (:require
   [ribelo.malmo :as ml]
   [ribelo.visby.math :as math])
  (:import
   (smile.regression
    ElasticNet
            ;; GaussianProcessRegression
    GradientTreeBoost$Trainer
    GradientTreeBoost$Loss
    LASSO
    OLS
    RidgeRegression
    RandomForest$Trainer
    SVR)
   (smile.math.kernel GaussianKernel)))

(comment
  (do (def train-x [[1 0 0] [2 1 0] [3 2 0] [4 3 0]])
      (def train-y [2 4 6 8])
      (def test-x  [[5 4 0]])))

(defn predict [model coll]
  (.predict model (ml/coll->x :double coll)))

(defn elastic-net
  ([train-x train-y ^double lambda-1 ^double lambda-2]
   (ElasticNet. (ml/coll->x :double train-x)
                (ml/coll->y :double train-y)
                lambda-1 lambda-2))
  ([train-x train-y lambda-1 lambda-2 tol max-iter]
   (ElasticNet. (ml/coll->x :double train-x)
                (ml/coll->y :double train-y)
                lambda-1 lambda-2 tol max-iter)))

(comment
  (predict (elastic-net train-x train-y 0.001 0.01) test-x))

(defn gaussian-process
  ([train-x train-y kernel lambda]
   (GaussianProcessRegression. (ml/coll->x :double train-x)
                               (ml/coll->y :double train-y)
                               kernel lambda)))

(comment
  (predict (gaussian-process train-x train-y (GaussianKernel. 0.06) 1.0) test-x))


(def ^:private gbm-loss
  {:huber                    GradientTreeBoost$Loss/Huber
   :least-absolute-deviation GradientTreeBoost$Loss/LeastAbsoluteDeviation
   :least-squares            GradientTreeBoost$Loss/LeastSquares})

(defn gradient-tree-boost
  ([train-x train-y]
   (gradient-tree-boost train-x train-y {:ntrees 500}))
  ([train-x train-y {:keys [^long ntrees loss ^long max-nodes
                            ^double shrinkage ^double f]}]
   (let [trainer (GradientTreeBoost$Trainer. ntrees)]
     (when loss      (.setLoss trainer (if (keyword? loss) (get gbm-loss loss) loss)))
     (when max-nodes (.setMaxNodes trainer max-nodes))
     (when shrinkage (.setShrinkage trainer shrinkage))
     (when f         (.setSamplingRates trainer f))
     (.train trainer (ml/coll->x :double train-x)
             (ml/coll->y :double train-y)))))

(comment
  (predict (gradient-tree-boost train-x train-y {:ntrees 500}) test-x))

(defn lasso
  ([train-x train-y ^double lambda]
   (LASSO. (ml/coll->x :double train-x)
           (ml/coll->y :double train-y)
           lambda))
  ([train-x train-y lambda tol max-iter]
   (LASSO. (ml/coll->x :double train-x)
           (ml/coll->y :double train-y)
           lambda tol max-iter)))

(comment
  (predict (lasso train-x train-y 0.01) test-x))

(defn ols
  ([train-x train-y]
   (ols train-x train-y true))
  ([train-x train-y svd?]
   (OLS. (ml/coll->x :double train-x)
         (ml/coll->y :double train-y)
         svd?)))

(comment
  (predict (ols train-x train-y) test-x))

(defn ridge
  [train-x train-y ^double lambda]
  (RidgeRegression. (ml/coll->x :double train-x)
                    (ml/coll->y :double train-y)
                    lambda))

(comment
  (predict (ridge-regression train-x train-y 0.01) test-x))

(defn random-forest
  ([train-x train-y]
   (random-forest train-x train-y {}))
  ([train-x train-y
    {:keys [ntrees max-nodes node-size mtry subsample]
     :or   {ntrees 500 node-size 5}}]
   (let [trainer (RandomForest$Trainer. ntrees)
         mtry'   (math/max 1.0 (if (nil? mtry) (math/floor (/ (count (first train-x)) 3)) mtry))]
     (when max-nodes (.setMaxNodes trainer max-nodes))
     (when node-size (.setNodeSize trainer node-size))
     (when mtry'     (.setNumRandomFeatures trainer mtry'))
     (when subsample (.setSamplingRates trainer subsample))
     (.train trainer
             (ml/coll->x :double train-x)
             (ml/coll->y :double train-y)))))

(comment
  (predict (random-forest train-x train-y) test-x))

(defn svr
  ([train-x train-y kernel eps c]
   (SVR. (ml/coll->x :double train-x)
         (ml/coll->y :double train-y)
         kernel eps c))
  ([train-x train-y kernel eps c toll]
   (SVR. (ml/coll->x :double train-x)
         (ml/coll->y :double train-y)
         kernel eps c toll)))

(comment
  (predict (svr train-x train-y (GaussianKernel. 0.06) 20 10) test-x))
