import tensorflow as tf



class FGM(tf.keras.Model):
    def __init__(self, *args, delta=0.2, eps=1e-4, start_step=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.start_step = start_step
        
    def train_step_fgm(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        embedding = self.trainable_variables[0]
        embedding_gradients = tape.gradient(loss, [self.trainable_variables[0]])[0]
        embedding_gradients = tf.zeros_like (embedding) + embedding_gradients
        delta = tf.math.divide_no_nan(self.delta * embedding_gradients , tf.math.sqrt(tf.reduce_sum(embedding_gradients**2)) + self.eps)
        self.trainable_variables[0].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
        self.trainable_variables[0].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        return tf.cond(self._train_counter < self.start_step, lambda:super(FGM, self).train_step(data), lambda:self.train_step_fgm(data))

        

class AWP(tf.keras.Model):
    def __init__(self, *args, delta=0.1, eps=1e-4, start_step=0,late_dropout =None,dropout_step = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.eps = eps
        self.late_dropout = late_dropout
        self.start_step = tf.constant(start_step, dtype=tf.int64)
        self.dropout_step = tf.constant(dropout_step, dtype=tf.int64)

        self.train_counter = tf.Variable(
              0,
              dtype=tf.int64,
              trainable=False
                )



    def train_step_awp(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        params = self.trainable_variables
        params_gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            self.trainable_variables[i].assign_add(delta)
        with tf.GradientTape() as tape2:
            y_pred = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
            
        gradients = tape2.gradient(new_loss, self.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            gradients =  self.optimizer.get_unscaled_gradients(gradients)
        for i in range(len(params_gradients)):
            grad = tf.zeros_like(params[i]) + params_gradients[i]
            delta = tf.math.divide_no_nan(self.delta * grad , tf.math.sqrt(tf.reduce_sum(grad**2)) + self.eps)
            self.trainable_variables[i].assign_sub(delta)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # self_loss.update_state(loss)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def train_step(self, data):
        x, y = data

        if self.late_dropout is not None:
            self.late_dropout.enabled.assign(
                tf.logical_or(self.late_dropout.enabled,
                              self.optimizer.iterations >= self.dropout_step)
            )

        mask = tf.cast(self.train_counter >= self.start_step, tf.float32)  # 0.0 or 1.0

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        params = self.trainable_variables
        base_grads = tape.gradient(loss, params)
        for i, g in enumerate(base_grads):
            if g is None:
                print(f"base_grads[{i}] is None")

        deltas = []
        for g, p in zip(base_grads, params):
            if g is None:
                g = tf.zeros_like(p)

            d = tf.math.divide_no_nan(
                self.delta * g,
                tf.math.sqrt(tf.reduce_sum(tf.square(g))) + self.eps
            )

            deltas.append(d * mask)

        for p, d in zip(params, deltas):
            p.assign_add(d)

        with tf.GradientTape() as tape2:
            y_pred2 = self(x, training=True)
            new_loss = self.compiled_loss(y, y_pred2, regularization_losses=self.losses)
            if hasattr(self.optimizer, 'get_scaled_loss'):
                new_loss = self.optimizer.get_scaled_loss(new_loss)
        
        awp_grads = tape2.gradient(new_loss, params)
        for i, g in enumerate(awp_grads):
            if g is None:
                print(f"awp_grads[{i}] is None")


        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            awp_grads = self.optimizer.get_unscaled_gradients(awp_grads)
        
        fixed_awp_grads = []
        for g, p in zip(awp_grads, params):
            if g is None:
                g = tf.zeros_like(p)
            fixed_awp_grads.append(g)

        awp_grads = fixed_awp_grads

        for p, d in zip(params, deltas):
            p.assign_sub(d)

        gradients = [
            mask * ag + (1.0 - mask) * bg
            for ag, bg in zip(awp_grads, base_grads)
        ]

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        self.train_counter.assign_add(1)
        return {m.name: m.result() for m in self.metrics}


        